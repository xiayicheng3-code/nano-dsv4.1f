from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..config import ModelConfig
from ..model import build_layer_specs
from .config_io import model_config_from_export, to_torch
from .decode_cache import NanoDecodeCache, attention_step
from .mhc_engram import mhc_mixes, post_mix, pre_mix
from .model_ops import apply_engram, apply_engram_step, apply_moe
from .rope_ops import linear, rms_norm, segment_local_positions
from .sparse_attention import TorchCSA2State, attention_forward


class NanoDeepseekV41CPU:
    """Correctness-first eager CPU runtime for portable nano-dsv4.1f weights.

    The runtime is intentionally separate from ``vllm.models.deepseek_v4``. It mirrors the
    V4.1/nano model semantics using ordinary Torch CPU operators. ``sparse_retrieval=False``
    matches the dense JAX training backbone, while ``True`` applies the trained Top-K indexer
    to global attention and reuses/reindexes selections according to the layer schedule.
    """

    def __init__(
        self,
        config: ModelConfig,
        flat_weights: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.weights = {
            key: to_torch(value, device=self.device, dtype=dtype)
            for key, value in flat_weights.items()
        }
        self.specs = build_layer_specs(config)
        self.max_context = 32768

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "NanoDeepseekV41CPU":
        root = Path(path)
        payload = json.loads(
            (root / "config.json").read_text(encoding="utf-8")
        )
        config = model_config_from_export(payload)
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "CPU inference requires `pip install 'nano-dsv41f[cpu]'`"
            ) from exc
        tensors = load_file(
            str(root / "model.safetensors"), device=str(device)
        )
        model = cls(config, tensors, device=device, dtype=dtype)
        model.max_context = int(payload.get("max_position_embeddings", 32768))
        return model

    def _w(self, path: str) -> torch.Tensor:
        key = path if path.startswith("nano.") else f"nano.{path}"
        try:
            return self.weights[key]
        except KeyError as exc:
            raise KeyError(
                f"portable checkpoint is missing tensor {key!r}"
            ) from exc

    def _norm(
        self,
        x: torch.Tensor,
        path: str,
        *,
        eps: float | None = None,
    ) -> torch.Tensor:
        return rms_norm(
            x,
            self._w(f"{path}.weight"),
            self.config.norm_eps if eps is None else eps,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        segment_ids: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        compute_indexer: bool = True,
        sparse_retrieval: bool = True,
        collect_draft_features: bool = False,
        logit_positions: torch.Tensor | None = None,
        return_layer_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        input_ids = input_ids.to(
            device=self.device, dtype=torch.long
        )
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch,tokens]")
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        else:
            segment_ids = segment_ids.to(
                device=self.device, dtype=torch.long
            )
        if segment_ids.shape != input_ids.shape:
            raise ValueError("segment_ids must match input_ids")
        if token_mask is not None:
            token_mask = token_mask.to(
                device=self.device, dtype=torch.bool
            )
            if token_mask.shape != input_ids.shape:
                raise ValueError("token_mask must match input_ids")
        if sparse_retrieval:
            compute_indexer = True

        streams = self._w("embed")[input_ids]
        streams = streams.unsqueeze(-2).expand(
            *streams.shape[:-1],
            self.config.mhc_streams,
            streams.shape[-1],
        ).clone()
        incoming_pre = torch.zeros(
            streams.shape[:-1],
            device=self.device,
            dtype=torch.float32,
        )
        incoming_pre[..., 0] = 1.0

        state: TorchCSA2State | None = None
        context_final: torch.Tensor | None = None
        layer_aux: list[dict[str, Any]] = []
        draft_features = []

        for spec in self.specs:
            layer_id = spec.layer_id
            global_source = None
            if spec.half == "generation" and context_final is None:
                context_final = pre_mix(streams, incoming_pre)
                global_source = context_final

            if self.config.engram.enabled and spec.has_engram:
                streams = apply_engram(
                    self,
                    streams,
                    input_ids,
                    segment_ids,
                    layer_id,
                    token_mask,
                )

            if collect_draft_features and layer_id in self.config.dspark.target_layer_ids:
                draft_features.append(streams.mean(dim=-2))

            residual = streams
            attn_mhc = f"blocks.{layer_id}.mhc_attn"
            attn_pre, attn_post, attn_comb = mhc_mixes(
                streams,
                self._w(f"{attn_mhc}.weight"),
                self._w(f"{attn_mhc}.base"),
                self._w(f"{attn_mhc}.scale"),
                sinkhorn_iters=self.config.mhc_sinkhorn_iters,
                eps=self.config.mhc_eps,
                norm_eps=self.config.norm_eps,
            )
            attn_input = self._norm(
                pre_mix(streams, incoming_pre),
                f"blocks.{layer_id}.attn_norm",
            )
            attn_out, state, aux = attention_forward(
                self,
                attn_input,
                segment_ids,
                layer_id,
                spec.mode,
                spec.owns_global_kv,
                spec.compression_ratio,
                state,
                global_source=global_source,
                compute_indexer=compute_indexer,
                sparse_retrieval=sparse_retrieval,
            )
            streams = post_mix(
                residual, attn_out, attn_comb, attn_post
            )

            residual = streams
            ffn_mhc = f"blocks.{layer_id}.mhc_ffn"
            ffn_pre, ffn_post, ffn_comb = mhc_mixes(
                streams,
                self._w(f"{ffn_mhc}.weight"),
                self._w(f"{ffn_mhc}.base"),
                self._w(f"{ffn_mhc}.scale"),
                sinkhorn_iters=self.config.mhc_sinkhorn_iters,
                eps=self.config.mhc_eps,
                norm_eps=self.config.norm_eps,
            )
            ffn_input = self._norm(
                pre_mix(streams, attn_pre),
                f"blocks.{layer_id}.ffn_norm",
            )
            ffn_out = apply_moe(self, ffn_input, layer_id)
            streams = post_mix(
                residual, ffn_out, ffn_comb, ffn_post
            )
            incoming_pre = ffn_pre
            if return_layer_aux:
                layer_aux.append(aux)

        hidden = self._norm(
            pre_mix(streams, incoming_pre), "final_norm"
        )
        selected_hidden = hidden
        if logit_positions is not None:
            positions = logit_positions.to(device=self.device, dtype=torch.long)
            if positions.ndim != 2 or positions.shape[0] != hidden.shape[0]:
                raise ValueError("logit_positions must have shape [batch, selected_tokens]")
            selected_hidden = hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        logits = linear(selected_hidden, self._w("lm_head"))
        return logits, {
            "final_hidden": hidden,
            "dspark_context_features": torch.cat(draft_features, dim=-1) if draft_features else None,
            "context_final": context_final,
            "layers": tuple(layer_aux),
            "final_global_source_layer": (
                None if state is None else state.source_layer
            ),
        }

    def new_cache(self, capacity: int | None = None) -> NanoDecodeCache:
        """Create an empty single-sequence cache for incremental CPU decoding."""
        return NanoDecodeCache.empty(device=self.device, capacity=self.max_context if capacity is None else capacity)

    @torch.inference_mode()
    def forward_chunk(
        self,
        token_ids: torch.Tensor,
        cache: NanoDecodeCache | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        compute_indexer: bool = True,
        sparse_retrieval: bool = True,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, NanoDecodeCache, dict[str, Any]]:
        """Process a causal token chunk in one target pass and update the cache."""
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        if token_ids.ndim == 0:
            token_ids = token_ids.reshape(1, 1)
        elif token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
            raise ValueError("incremental CPU reference currently supports batch size 1")
        if cache is None:
            cache = self.new_cache()
        if cache.input_ids.shape[0] != 1:
            raise ValueError("incremental CPU reference currently supports batch size 1")

        if segment_ids is None:
            current_segment = (
                torch.zeros_like(token_ids)
                if cache.length == 0
                else cache.segment_ids[:, -1:].expand_as(token_ids)
            )
        else:
            current_segment = segment_ids.to(device=self.device, dtype=torch.long)
            if current_segment.ndim == 0:
                current_segment = current_segment.reshape(1, 1)
            elif current_segment.ndim == 1:
                current_segment = current_segment.unsqueeze(0)
            if current_segment.shape != token_ids.shape:
                raise ValueError("segment_ids must match the token chunk")

        if sparse_retrieval:
            compute_indexer = True
        mode = (compute_indexer, sparse_retrieval)
        if cache.mode is not None and cache.mode != mode:
            raise ValueError("attention mode changed; start a fresh decode cache")
        cache.mode = mode
        position = cache.append_tokens(token_ids, current_segment)

        streams = self._w("embed")[token_ids]
        streams = streams.unsqueeze(-2).expand(
            1, token_ids.shape[1], self.config.mhc_streams, self.config.d_model
        ).clone()
        incoming_pre = torch.zeros(
            (1, token_ids.shape[1], self.config.mhc_streams),
            device=self.device,
            dtype=torch.float32,
        )
        incoming_pre[..., 0] = 1.0

        state: TorchCSA2State | None = None
        context_final: torch.Tensor | None = None
        layer_aux: list[dict[str, Any]] = []
        draft_features = []

        for spec in self.specs:
            layer_id = spec.layer_id
            global_source = None
            if spec.half == "generation" and context_final is None:
                context_final = pre_mix(streams, incoming_pre)
                global_source = context_final

            if self.config.engram.enabled and spec.has_engram:
                streams = apply_engram_step(
                    self,
                    streams,
                    cache.input_ids,
                    cache.segment_ids,
                    layer_id,
                )

            if cache.collect_draft and layer_id in self.config.dspark.target_layer_ids:
                draft_features.append(streams.mean(dim=-2))

            residual = streams
            attn_mhc = f"blocks.{layer_id}.mhc_attn"
            attn_pre, attn_post, attn_comb = mhc_mixes(
                streams,
                self._w(f"{attn_mhc}.weight"),
                self._w(f"{attn_mhc}.base"),
                self._w(f"{attn_mhc}.scale"),
                sinkhorn_iters=self.config.mhc_sinkhorn_iters,
                eps=self.config.mhc_eps,
                norm_eps=self.config.norm_eps,
            )
            attn_input = self._norm(
                pre_mix(streams, incoming_pre),
                f"blocks.{layer_id}.attn_norm",
            )
            attn_out, state, aux = attention_step(
                self,
                cache,
                attn_input,
                layer_id=layer_id,
                mode=spec.mode,
                owns_global_kv=spec.owns_global_kv,
                compression_ratio=spec.compression_ratio,
                state=state,
                segment=current_segment,
                position=position,
                global_source=global_source,
                compute_indexer=compute_indexer,
                sparse_retrieval=sparse_retrieval,
            )
            streams = post_mix(
                residual, attn_out, attn_comb, attn_post
            )

            residual = streams
            ffn_mhc = f"blocks.{layer_id}.mhc_ffn"
            ffn_pre, ffn_post, ffn_comb = mhc_mixes(
                streams,
                self._w(f"{ffn_mhc}.weight"),
                self._w(f"{ffn_mhc}.base"),
                self._w(f"{ffn_mhc}.scale"),
                sinkhorn_iters=self.config.mhc_sinkhorn_iters,
                eps=self.config.mhc_eps,
                norm_eps=self.config.norm_eps,
            )
            ffn_input = self._norm(
                pre_mix(streams, attn_pre),
                f"blocks.{layer_id}.ffn_norm",
            )
            ffn_out = apply_moe(self, ffn_input, layer_id)
            streams = post_mix(
                residual, ffn_out, ffn_comb, ffn_post
            )
            incoming_pre = ffn_pre
            layer_aux.append(aux)

        if draft_features:
            from .dspark import record_context
            record_context(self, cache, torch.cat(draft_features, dim=-1), position)
        hidden = self._norm(
            pre_mix(streams, incoming_pre), "final_norm"
        )
        logits = linear(hidden if return_all_logits else hidden[:, -1:], self._w("lm_head"))
        cache.next_logits = logits[:, -1].float().clone()
        return logits, cache, {
            "final_hidden": hidden,
            "context_final": context_final,
            "layers": tuple(layer_aux),
            "final_global_source_layer": (
                None if state is None else state.source_layer
            ),
        }

    @torch.inference_mode()
    def forward_step(self, token_ids, cache=None, **kwargs):
        """Single-token compatibility entry point using the shared chunk operators."""
        token_ids = torch.as_tensor(token_ids, device=self.device, dtype=torch.long)
        if token_ids.numel() != 1:
            raise ValueError("forward_step requires one token; use forward_chunk")
        return self.forward_chunk(token_ids.reshape(1, 1), cache, **kwargs)

    @torch.inference_mode()
    def prefill_cache(
        self,
        input_ids: torch.Tensor,
        *,
        segment_ids: torch.Tensor | None = None,
        compute_indexer: bool = True,
        sparse_retrieval: bool = True,
        cache: NanoDecodeCache | None = None,
        return_all_logits: bool = True,
        chunk_size: int = 32,
    ) -> tuple[torch.Tensor, NanoDecodeCache]:
        """Prefill in bounded causal chunks; chunk_size=1 is the scalar baseline."""
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        ids = input_ids.to(device=self.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("prefill_cache requires a non-empty [1,tokens] input")
        if segment_ids is None:
            segments = (torch.zeros_like(ids) if cache is None or cache.length == 0
                        else cache.segment_ids[:, -1:].expand_as(ids))
        else:
            segments = segment_ids.to(device=self.device, dtype=torch.long)
            if segments.shape != ids.shape:
                raise ValueError("segment_ids must match input_ids")

        cache = self.new_cache() if cache is None else cache
        if cache.length + ids.shape[1] > cache.capacity:
            raise ValueError("prefill exceeds cache capacity")
        rows: list[torch.Tensor] = []
        for position in range(0, ids.shape[1], chunk_size):
            logits, cache, _ = self.forward_chunk(
                ids[:, position : position + chunk_size],
                cache,
                segment_ids=segments[:, position : position + chunk_size],
                compute_indexer=compute_indexer,
                sparse_retrieval=sparse_retrieval,
                return_all_logits=return_all_logits,
            )
            if return_all_logits:
                rows.append(logits)
        return (torch.cat(rows, dim=1) if return_all_logits else logits), cache

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_p >= 1.0:
            return torch.multinomial(probs, num_samples=1)
        sorted_probs, sorted_idx = torch.sort(
            probs, descending=True, dim=-1
        )
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(
            dim=-1, keepdim=True
        )
        sample = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, sample)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        sparse_retrieval: bool = True,
    ) -> torch.Tensor:
        """Stateless convenience API; use InferenceSession for prefix reuse/statistics."""
        from .session import InferenceSession
        return InferenceSession(self).generate(
            input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
            temperature=temperature, top_p=top_p, sparse_retrieval=sparse_retrieval)
