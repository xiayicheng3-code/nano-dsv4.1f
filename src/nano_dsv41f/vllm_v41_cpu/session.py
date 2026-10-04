"""Bounded, single-session prefix caching and explicit phase measurements."""
from __future__ import annotations

from threading import RLock
from time import perf_counter

import torch


class InferenceSession:
    """Retain one conversation in RAM; do not share this object across users.

    A prompt checkpoint permits reuse when protocol rendering rewrites the reply.
    Cache reset drops all state. We never persist conversations or keys to disk.
    """

    def __init__(self, model, *, capacity=None, mtp=False, draft_trained=False,
                 allow_untrained_draft=False):
        self.model = model
        self.capacity = model.max_context if capacity is None else capacity
        if not 0 < self.capacity <= model.max_context:
            raise ValueError("capacity must fit the exported model context")
        if mtp and not model.config.dspark.enabled:
            raise ValueError("this checkpoint has no DSpark head")
        if mtp and not (draft_trained or allow_untrained_draft):
            raise ValueError("DSpark training is unverified; current training freezes it. "
                             "Set allow_untrained_draft=True only for diagnostics.")
        self.mtp = mtp
        self._lock = RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.cache = None
            self.prompt_checkpoint = None
            self.last_stats = {}

    def _clock(self):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        return perf_counter()

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens, eos_token_id=None,
                 temperature=0.0, top_p=1.0, sparse_retrieval=True):
        with self._lock:
            try:
                return self._generate(input_ids, max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id, temperature=temperature, top_p=top_p,
                    sparse_retrieval=sparse_retrieval)
            except BaseException:
                # A partially updated cache must never serve another request.
                self.reset()
                raise

    def _generate(self, input_ids, *, max_new_tokens, eos_token_id,
                  temperature, top_p, sparse_retrieval):
        ids = input_ids.to(device=self.model.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("generate requires a non-empty single-sequence prompt")
        if max_new_tokens < 0 or not 0 < top_p <= 1 or temperature < 0:
            raise ValueError("invalid generation budget, temperature, or top_p")
        if ids.shape[1] + max_new_tokens > self.capacity:
            raise ValueError("prompt plus output budget exceeds cache capacity")
        if self.mtp and temperature != 0:
            raise ValueError("experimental MTP currently verifies greedy decoding only; set temperature=0")
        started = self._clock()
        self.last_stats = {}
        mode = (True, sparse_retrieval)
        def matches(length):
            return length <= ids.shape[1] and torch.equal(
                self.cache.input_ids[:, :length], ids[:, :length])
        if self.cache is not None:
            if self.cache.mode != mode:
                self.cache = None
            elif not matches(self.cache.length):
                saved = self.prompt_checkpoint
                if saved is not None and matches(saved["length"]):
                    self.cache.restore(saved)
                else:
                    self.cache = None
        if self.cache is None:
            self.cache = self.model.new_cache(self.capacity)
            self.cache.collect_draft = self.mtp
        reused = self.cache.length
        prompt_tokens = ids.shape[1]
        prefill_start = self._clock()
        if reused < prompt_tokens:
            self.model.prefill_cache(ids[:, reused:], cache=self.cache,
                sparse_retrieval=sparse_retrieval, return_all_logits=False)
        prefill_end = self._clock()
        self.prompt_checkpoint = self.cache.snapshot()
        decode_start = self._clock()
        output = []
        per_token = []
        proposed = accepted = verified = 0
        draft_seconds = 0.0
        ttft = None
        stopped = False
        while len(output) < max_new_tokens:
            proposals = []
            if self.mtp:
                from .dspark import draft_block
                draft_start = self._clock()
                proposals = draft_block(self.model, self.cache)[0].tolist()
                proposals = proposals[:max_new_tokens - len(output)]
                draft_seconds += self._clock() - draft_start
                proposed += len(proposals)
            for i in range(len(proposals) or 1):
                token_start = self._clock()
                # Sequential target verification is deliberately conservative. It
                # preserves greedy output but does not yet amortize target forwards.
                next_id = self.model._sample_next(self.cache.next_logits,
                    temperature=temperature, top_p=top_p)
                token = int(next_id.item())
                match = bool(proposals) and token == proposals[i]
                if proposals:
                    verified += 1
                    accepted += int(match)
                output.append(token)
                if ttft is None:
                    ttft = self._clock() - started
                stopped = token == eos_token_id
                if not stopped and len(output) < max_new_tokens:
                    self.model.forward_step(next_id, self.cache,
                        sparse_retrieval=sparse_retrieval)
                per_token.append(self._clock() - token_start)
                if stopped or len(output) == max_new_tokens or not match:
                    break
            if stopped:
                break
        ended = self._clock()
        prefill_seconds = prefill_end - prefill_start
        decode_seconds = ended - decode_start
        def rate(count, seconds):
            return count / seconds if count and seconds > 0 else 0.0
        # Count unique tensor allocations, not overlapping views.
        buffers = [self.cache.input_buffer, self.cache.segment_buffer, self.cache.position_buffer]
        for name in ("local_kv", "local_segments", "local_positions"):
            buffers.extend(getattr(self.cache, name).values())
        for owner in self.cache.owners.values():
            buffers.extend(owner.buffers.values())
        for name in ("draft_kv", "draft_positions"):
            if getattr(self.cache, name) is not None:
                buffers.append(getattr(self.cache, name))
        self.last_stats = dict(
            prompt_tokens=prompt_tokens, cached_prompt_tokens=reused,
            prefill_tokens=prompt_tokens-reused, prefill_seconds=prefill_seconds,
            prefill_tokens_per_second=rate(prompt_tokens-reused, prefill_seconds),
            decode_tokens=len(output), decode_seconds=decode_seconds,
            decode_tokens_per_second=rate(len(output), decode_seconds),
            time_to_first_token_seconds=ttft, total_seconds=ended-started,
            decode_step_seconds=per_token, cache_capacity=self.capacity,
            cache_tokens=self.cache.length,
            cache_arena_bytes=sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in buffers}.values()),
            mtp_proposed_tokens=proposed, mtp_verified_tokens=verified,
            mtp_accepted_tokens=accepted,
            mtp_acceptance_rate=accepted/verified if verified else None,
            mtp_draft_seconds=draft_seconds,
            mtp_verifier="sequential_greedy" if self.mtp else "disabled",
        )
        suffix = ids.new_tensor([output])
        return torch.cat((ids, suffix), dim=1)


def format_inference_stats(stats):
    """Compact per-turn measurements for notebook and terminal clients."""
    if not stats:
        return ""
    lines = [
        f"Prefill: {stats['prefill_tokens']} new tokens, "
        f"{stats['prefill_seconds']:.3f} s, {stats['prefill_tokens_per_second']:.2f} tok/s "
        f"({stats['cached_prompt_tokens']}/{stats['prompt_tokens']} prompt tokens reused)",
        f"Decode: {stats['decode_tokens']} tokens, {stats['decode_seconds']:.3f} s, "
        f"{stats['decode_tokens_per_second']:.2f} tok/s",
        f"Total: {stats['total_seconds']:.3f} s; "
        f"cache arena: {stats['cache_arena_bytes']/2**20:.2f} MiB "
        f"({stats['cache_tokens']}/{stats['cache_capacity']} tokens)",
    ]
    if stats['time_to_first_token_seconds'] is not None:
        lines.append(f"First token: {stats['time_to_first_token_seconds']:.3f} s")
    if stats['mtp_verified_tokens']:
        lines.append(f"MTP: {stats['mtp_accepted_tokens']}/{stats['mtp_verified_tokens']} verified "
                     f"proposals accepted; {stats['mtp_proposed_tokens']} proposed; "
                     f"draft time {stats['mtp_draft_seconds']:.3f} s (sequential verifier)")
    return "\n".join(lines)
