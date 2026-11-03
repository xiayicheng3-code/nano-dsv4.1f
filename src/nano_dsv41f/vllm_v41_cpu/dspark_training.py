"""Frozen-target DSpark distillation using the same Torch operators as serving."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
import torch.nn.functional as F

from .dspark import draft_from_features


@dataclass(frozen=True)
class DraftLossConfig:
    ce_weight: float = 0.1
    distribution_weight: float = 0.9
    confidence_weight: float = 1.0

    def __post_init__(self):
        if any(not 0 <= x < float('inf') for x in asdict(self).values()):
            raise ValueError('Loss weights must be finite and nonnegative')
        if not any(asdict(self).values()):
            raise ValueError('At least one draft loss must be enabled')


def draft_loss(logits, target_logits, labels, confidence_logits, *, config=DraftLossConfig()):
    """Paper equations 8–12, averaged with normalized position-decay weights.

    The paper names the full L1 distance L_tv (twice mathematical TV). Keep
    that factor here. Confidence labels and all teacher outputs are detached.
    """
    target = torch.softmax(target_logits.detach().float(), dim=-1)
    log_draft = F.log_softmax(logits.float(), dim=-1)
    draft = log_draft.exp()
    ce = -log_draft.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    l1 = (draft - target).abs().sum(-1)
    acceptance = (1.0 - 0.5 * l1.detach()).clamp(0, 1)
    confidence = F.binary_cross_entropy_with_logits(confidence_logits.float(), acceptance, reduction='none')
    width = logits.shape[-2]
    position_weight = torch.exp(-torch.arange(width, device=logits.device).float() / width)
    weight = position_weight[None].expand_as(ce)
    def mean(value):
        return (value * weight).sum() / weight.sum()
    ce_mean, l1_mean, confidence_mean = map(mean, (ce, l1, confidence))
    loss = config.ce_weight * ce_mean + config.distribution_weight * l1_mean + config.confidence_weight * confidence_mean
    with torch.no_grad():
        agreement = logits.argmax(-1) == target_logits.argmax(-1)
        metrics = dict(loss=loss.detach(), ce=ce_mean.detach(), distribution_l1=l1_mean.detach(),
            confidence_bce=confidence_mean.detach(), teacher_forced_overlap=acceptance.mean(),
            teacher_forced_top1_agreement=agreement.float().mean(),
            confidence_mae=(confidence_logits.sigmoid()-acceptance).abs().mean(),
            tokens=labels.numel())
        for i in range(width):
            metrics[f'position_{i+1}_overlap'] = acceptance[:, i].mean()
            metrics[f'position_{i+1}_top1'] = agreement[:, i].float().mean()
    return loss, metrics


def backbone_digest(model):
    """Hash every frozen runtime tensor, including embeddings and the LM head."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.weights.items()):
        if name.startswith('nano.dspark.'):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class DraftTrainer:
    def __init__(self, model, *, learning_rate=1e-4, loss_config=DraftLossConfig()):
        if not model.config.dspark.enabled:
            raise ValueError('Checkpoint has no DSpark parameters')
        if model.dtype != torch.float32:
            raise ValueError('Draft training currently uses FP32 for numerical stability')
        if not model.config.dspark.confidence_head and loss_config.confidence_weight:
            raise ValueError('Confidence loss requires an enabled confidence head')
        self.model, self.loss_config = model, loss_config
        self.parameters = []
        for name, value in list(model.weights.items()):
            # Router bias uses the separate load-balancing controller, not Adam.
            if name.startswith('nano.dspark.') and not name.endswith('router_bias'):
                value = torch.nn.Parameter(value.detach().clone())
                model.weights[name] = value
                self.parameters.append(value)
            else:
                value.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate, weight_decay=0.01)
        self.steps = 0

    def prepare(self, input_ids, anchors):
        """One unpadded conversation crop; no cross-document context is admitted."""
        ids = torch.as_tensor(input_ids, device=self.model.device, dtype=torch.long).reshape(1, -1)
        anchors = torch.as_tensor(anchors, device=self.model.device, dtype=torch.long).reshape(-1)
        width = self.model.config.dspark.block_size
        if not anchors.numel() or int(anchors.min()) < 0 or int(anchors.max()) + width >= ids.shape[1]:
            raise ValueError('Every anchor must have a complete in-bounds draft block')
        positions = anchors[:, None] + torch.arange(width, device=self.model.device)[None]
        # Gather target vocabulary logits only at the selected block positions.
        teacher_logits, aux = self.model.forward(ids, sparse_retrieval=True,
            collect_draft_features=True, logit_positions=positions.reshape(1, -1),
            return_layer_aux=False)
        window = self.model.config.attention.local_window
        context_positions = anchors[:, None] - torch.arange(window-1, -1, -1, device=self.model.device)[None]
        context_mask = context_positions >= 0
        context_positions = context_positions.clamp_min(0)
        # Clone outside inference_mode: these constants will be saved for draft backward.
        features = aux['dspark_context_features'][0, context_positions].clone().detach()
        return dict(features=features, context_positions=context_positions, context_mask=context_mask,
            anchor_ids=ids[0, anchors].clone(), anchor_positions=anchors,
            previous_ids=ids[0, positions].clone(), labels=ids[0, positions+1].clone(),
            target_logits=teacher_logits.reshape(len(anchors), width, -1).clone().detach())

    def loss(self, batch):
        logits, confidence, routes = draft_from_features(self.model, batch['features'],
            batch['context_positions'], batch['context_mask'], batch['anchor_ids'],
            batch['anchor_positions'], batch['previous_ids'], return_router_indices=True)
        loss, metrics = draft_loss(logits, batch['target_logits'], batch['labels'], confidence,
                          config=self.loss_config)
        metrics["router_loads"] = torch.bincount(routes.detach().reshape(-1),
            minlength=self.model.config.dspark.n_routed_experts).float() / routes.numel()
        return loss, metrics

    def train_step(self, input_ids, anchors):
        batch = self.prepare(input_ids, anchors)
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite draft loss; no update applied')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        loads = metrics.pop('router_loads')
        with torch.no_grad():
            self.model._w('dspark.moe.router_bias').add_(
                self.model.config.router_bias_update_speed * torch.sign(loads.mean()-loads))
        self.steps += 1
        metrics['gradient_norm'] = norm.detach()
        return {name: float(value) for name, value in metrics.items()}

    @torch.no_grad()
    def evaluate(self, input_ids, anchors):
        _, metrics = self.loss(self.prepare(input_ids, anchors))
        loads = metrics.pop('router_loads')
        metrics.update({f'expert_{i}_fraction': value for i,value in enumerate(loads)})
        return {name: float(value) for name, value in metrics.items()}

    def draft_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.model.weights.items()
                if name.startswith('nano.dspark.')}

    def load_draft(self, state):
        expected = {name for name in self.model.weights if name.startswith('nano.dspark.')}
        if set(state) != expected:
            raise ValueError('Draft checkpoint tensor names do not match')
        for name, value in state.items():
            if value.shape != self.model.weights[name].shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid draft tensor: {name}')
        with torch.no_grad():
            for name, value in state.items():
                self.model.weights[name].copy_(value.to(self.model.device))
