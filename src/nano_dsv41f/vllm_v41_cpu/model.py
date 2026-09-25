from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..config import ModelConfig
from ..model import build_layer_specs
from .config_io import model_config_from_export, to_torch
from .mhc_engram import mhc_mixes, post_mix, pre_mix
from .model_ops import apply_engram, apply_moe
from .rope_ops import linear, rms_norm
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
        return cls(config, tensors, device=device, dtype=dtype)

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
            layer_aux.append(aux)

        hidden = self._norm(
            pre_mix(streams, incoming_pre), "final_norm"
        )
        logits = linear(hidden, self._w("lm_head"))
        return logits, {
            "final_hidden": hidden,
            "context_final": context_final,
            "layers": tuple(layer_aux),
            "final_global_source_layer": (
                None if state is None else state.source_layer
            ),
        }

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
        """Correctness path that recomputes the prefix before paged-cache integration."""
        ids = input_ids.to(
            device=self.device, dtype=torch.long
        )
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        for _ in range(max_new_tokens):
            logits, _ = self.forward(
                ids, sparse_retrieval=sparse_retrieval
            )
            next_logits = logits[:, -1].float()
            if temperature <= 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(
                    next_logits / temperature, dim=-1
                )
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(
                        probs, descending=True, dim=-1
                    )
                    cumulative = sorted_probs.cumsum(dim=-1)
                    remove = cumulative - sorted_probs > top_p
                    sorted_probs = sorted_probs.masked_fill(
                        remove, 0.0
                    )
                    sorted_probs = sorted_probs / sorted_probs.sum(
                        dim=-1, keepdim=True
                    )
                    sample = torch.multinomial(
                        sorted_probs, num_samples=1
                    )
                    next_id = sorted_idx.gather(-1, sample)
                else:
                    next_id = torch.multinomial(
                        probs, num_samples=1
                    )
            ids = torch.cat((ids, next_id), dim=-1)
            if (
                eos_token_id is not None
                and torch.all(next_id == eos_token_id)
            ):
                break
        return ids
