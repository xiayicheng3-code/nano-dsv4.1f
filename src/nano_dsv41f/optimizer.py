from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class OptimizerLeafState(NamedTuple):
    momentum: jax.Array
    second: jax.Array


def _path_parts(path) -> tuple[str, ...]:
    parts: list[str] = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        elif hasattr(key, "name"):
            parts.append(str(key.name))
        else:
            parts.append(str(key))
    return tuple(parts)


def classify_parameter(path, value: jax.Array, config) -> str:
    """Return the inspectable V4.1 optimizer family for one parameter leaf."""
    parts = _path_parts(path)
    if parts and parts[0] in {"embed", "lm_head"}:
        return "sinkhorn"
    if "engram" in parts and "table" in parts:
        return "sinkhorn"

    adam_names = {
        "router_bias",
        "attn_sink",
        "base",
        "scale",
        "q_weight",
        "k_weight",
    }
    norm_markers = {
        "attn_norm",
        "ffn_norm",
        "q_norm",
        "local_kv_norm",
        "global_norm",
        "k_norm",
        "kv_norm",
        "main_norm",
        "final_norm",
        "norm",
    }
    if (
        value.ndim < 2
        or any(p in adam_names for p in parts)
        or any(p in norm_markers for p in parts)
    ):
        return "adamw"

    if config.optimizer.headwise_qk_muon:
        # In the language MLA path the explicitly head-concatenated matrix is Q. The
        # latent K projection is shared rather than laid out by attention head, so we do
        # not invent a fake head axis merely to satisfy the optimizer label.
        if "q_b" in parts or ("indexer" in parts and "wq_b" in parts):
            return "headwise_muon"
    return "muon"


def parameter_rule_map(params, config) -> dict[str, str]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {
        "/".join(_path_parts(path)): classify_parameter(path, value, config)
        for path, value in leaves
    }


def init_optimizer_state(params, config):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    states = []
    for path, value in leaves:
        rule = classify_parameter(path, value, config)
        momentum = jnp.zeros_like(value, dtype=jnp.float32)
        second = (
            jnp.zeros_like(value, dtype=jnp.float32)
            if rule == "adamw"
            else jnp.empty((0,), dtype=jnp.float32)
        )
        states.append(OptimizerLeafState(momentum, second))
    return treedef.unflatten(states)


def _ns_matrix(x: jax.Array, fast_steps: int, stable_steps: int) -> jax.Array:
    if x.ndim != 2:
        raise ValueError("_ns_matrix expects a matrix")
    transposed = x.shape[0] > x.shape[1]
    y = x.T if transposed else x
    y = y / jnp.maximum(jnp.linalg.norm(y), 1e-7)

    def step(y, coeff):
        a, b, c = coeff
        gram = y @ y.T
        return a * y + b * (gram @ y) + c * ((gram @ gram) @ y)

    for _ in range(fast_steps):
        y = step(y, (3.4445, -4.7750, 2.0315))
    for _ in range(stable_steps):
        y = step(y, (2.0, -1.5, 0.5))
    return y.T if transposed else y


def hybrid_newton_schulz(
    x: jax.Array,
    *,
    fast_steps: int = 8,
    stable_steps: int = 2,
) -> jax.Array:
    if x.ndim < 2:
        raise ValueError("Muon requires matrix-valued parameters")
    if x.ndim == 2:
        return _ns_matrix(x, fast_steps, stable_steps)
    batch = x.reshape((-1,) + x.shape[-2:])
    out = jax.vmap(lambda m: _ns_matrix(m, fast_steps, stable_steps))(batch)
    return out.reshape(x.shape)


def _muon_update(
    param: jax.Array,
    grad: jax.Array,
    state: OptimizerLeafState,
    *,
    lr: jax.Array,
    config,
) -> tuple[jax.Array, OptimizerLeafState]:
    oc = config.optimizer
    m = oc.muon_momentum * state.momentum + grad.astype(jnp.float32)
    nesterov = oc.muon_momentum * m + grad.astype(jnp.float32)
    direction = hybrid_newton_schulz(
        nesterov,
        fast_steps=oc.muon_fast_steps,
        stable_steps=oc.muon_stable_steps,
    )
    scale = jnp.sqrt(float(max(param.shape[-2:]))) * oc.muon_update_rms
    updated = (
        param.astype(jnp.float32) * (1.0 - lr * oc.muon_weight_decay)
        - lr * direction * scale
    )
    return updated.astype(param.dtype), OptimizerLeafState(m, state.second)


def _headwise_muon_update(
    param: jax.Array,
    grad: jax.Array,
    state: OptimizerLeafState,
    *,
    lr: jax.Array,
    n_heads: int,
    head_dim: int,
    config,
) -> tuple[jax.Array, OptimizerLeafState]:
    if param.ndim != 2 or param.shape[-1] != n_heads * head_dim:
        return _muon_update(param, grad, state, lr=lr, config=config)
    oc = config.optimizer
    m = oc.muon_momentum * state.momentum + grad.astype(jnp.float32)
    nesterov = oc.muon_momentum * m + grad.astype(jnp.float32)
    heads = nesterov.reshape(param.shape[0], n_heads, head_dim).transpose(1, 0, 2)
    direction = hybrid_newton_schulz(
        heads,
        fast_steps=oc.muon_fast_steps,
        stable_steps=oc.muon_stable_steps,
    ).transpose(1, 0, 2).reshape(param.shape)
    scale = jnp.sqrt(float(max(param.shape[0], head_dim))) * oc.muon_update_rms
    updated = (
        param.astype(jnp.float32) * (1.0 - lr * oc.muon_weight_decay)
        - lr * direction * scale
    )
    return updated.astype(param.dtype), OptimizerLeafState(m, state.second)


def _adamw_update(
    param: jax.Array,
    grad: jax.Array,
    state: OptimizerLeafState,
    *,
    lr: jax.Array,
    step: jax.Array,
    config,
) -> tuple[jax.Array, OptimizerLeafState]:
    oc = config.optimizer
    g = grad.astype(jnp.float32)
    m = oc.adam_beta1 * state.momentum + (1.0 - oc.adam_beta1) * g
    v = oc.adam_beta2 * state.second + (1.0 - oc.adam_beta2) * jnp.square(g)
    t = step.astype(jnp.float32) + 1.0
    mhat = m / (1.0 - oc.adam_beta1**t)
    vhat = v / (1.0 - oc.adam_beta2**t)
    direction = mhat / (jnp.sqrt(vhat) + oc.adam_eps)
    updated = (
        param.astype(jnp.float32) * (1.0 - lr * oc.adam_weight_decay)
        - lr * direction
    )
    return updated.astype(param.dtype), OptimizerLeafState(m, v)


def sinkhorn_balance(
    x: jax.Array,
    *,
    iters: int,
    eps: float,
    row_mask_tau: float,
) -> jax.Array:
    """V4.1 Sinkhorn balancing: K *alternating* row/column L2 normalizations."""
    if x.ndim != 2:
        raise ValueError("Sinkhorn-balanced optimizer is defined for matrices")
    row_norm = jnp.linalg.norm(x, axis=1, keepdims=True)
    mean_row_norm = jnp.mean(row_norm)
    live = row_norm > row_mask_tau * mean_row_norm
    y = jnp.where(live, x, 0.0)
    for k in range(1, iters + 1):
        if k % 2 == 1:
            y = y / (jnp.linalg.norm(y, axis=1, keepdims=True) + eps)
            y = jnp.where(live, y, 0.0)
        else:
            y = y / (jnp.linalg.norm(y, axis=0, keepdims=True) + eps)
    return jnp.where(live, y, 0.0)


def _sinkhorn_update(
    param: jax.Array,
    grad: jax.Array,
    state: OptimizerLeafState,
    *,
    lr: jax.Array,
    config,
    transpose_for_rows: bool = False,
) -> tuple[jax.Array, OptimizerLeafState]:
    """Disclosed V4.1 momentum + Nesterov + Sinkhorn-balanced matrix update.

    `transpose_for_rows=True` is used for this repo's `[D,V]` LM-head storage so the
    semantic Sinkhorn rows are vocabulary entries `[V,D]`, just like token/Engram tables.
    """
    oc = config.optimizer
    beta = oc.sinkhorn_momentum
    g = grad.astype(jnp.float32)
    m = beta * state.momentum + (1.0 - beta) * g
    nesterov = beta * m + (1.0 - beta) * g

    semantic = nesterov.T if transpose_for_rows else nesterov
    balanced = sinkhorn_balance(
        semantic,
        iters=oc.sinkhorn_iters,
        eps=oc.sinkhorn_eps,
        row_mask_tau=oc.sinkhorn_row_mask_tau,
    )
    # Algorithm: Delta = sqrt(n_columns) U^(K), W -= eta * gamma * Delta.
    delta = jnp.sqrt(float(semantic.shape[1])) * balanced
    if transpose_for_rows:
        delta = delta.T
    updated = param.astype(jnp.float32) - lr * oc.sinkhorn_gamma * delta
    return updated.astype(param.dtype), OptimizerLeafState(m, state.second)


def learning_rate(step: jax.Array, train_config) -> jax.Array:
    """Configurable linear warmup, plateau, then late cosine decay."""
    s = step.astype(jnp.float32)
    peak = jnp.float32(train_config.learning_rate)
    minimum = jnp.float32(train_config.min_learning_rate)
    warm = peak * jnp.minimum((s + 1.0) / max(train_config.warmup_steps, 1), 1.0)
    decay_start = train_config.total_steps * train_config.cosine_decay_start_fraction
    decay_progress = jnp.clip(
        (s - decay_start) / max(train_config.total_steps - decay_start, 1.0),
        0.0,
        1.0,
    )
    cosine = minimum + 0.5 * (peak - minimum) * (
        1.0 + jnp.cos(jnp.pi * decay_progress)
    )
    return jnp.where(s < train_config.warmup_steps, warm, cosine)


def optimizer_step(params, grads, state, *, step: jax.Array, config, train_config):
    p_items, treedef = jax.tree_util.tree_flatten_with_path(params)
    g_leaves = jax.tree_util.tree_leaves(grads)
    s_leaves = jax.tree_util.tree_leaves(
        state, is_leaf=lambda x: isinstance(x, OptimizerLeafState)
    )
    if not (len(p_items) == len(g_leaves) == len(s_leaves)):
        raise ValueError("params/grads/optimizer state pytrees do not align")

    lr = learning_rate(step, train_config)
    new_params = []
    new_states = []
    for (path, param), grad, leaf_state in zip(p_items, g_leaves, s_leaves):
        parts = _path_parts(path)
        rule = classify_parameter(path, param, config)
        if rule == "adamw":
            p, s = _adamw_update(
                param, grad, leaf_state, lr=lr, step=step, config=config
            )
        elif rule == "sinkhorn":
            p, s = _sinkhorn_update(
                param,
                grad,
                leaf_state,
                lr=lr,
                config=config,
                transpose_for_rows=bool(parts and parts[0] == "lm_head"),
            )
        elif rule == "headwise_muon":
            if "indexer" in parts:
                nh, hd = config.indexer.n_heads, config.indexer.head_dim
            else:
                nh, hd = config.attention.n_heads, config.attention.head_dim
            p, s = _headwise_muon_update(
                param,
                grad,
                leaf_state,
                lr=lr,
                n_heads=nh,
                head_dim=hd,
                config=config,
            )
        else:
            p, s = _muon_update(param, grad, leaf_state, lr=lr, config=config)
        new_params.append(p)
        new_states.append(s)
    return (
        treedef.unflatten(new_params),
        treedef.unflatten(new_states),
        {"learning_rate": lr},
    )
