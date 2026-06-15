"""Generate a small, controlled experiment set from retrieved strategy cards."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from knowledge.schemas import ExperimentProposal


SUPPORTED_EXPERIMENT_FIELDS = {
    "augmentation",
    "randaugment_num_ops",
    "randaugment_magnitude",
    "mixup_alpha",
    "cutmix_alpha",
    "label_smoothing",
    "optimizer",
    "scheduler",
    "learning_rate",
    "backbone_learning_rate",
    "head_learning_rate",
    "finetune_strategy",
    "freeze_backbone",
    "unfreeze_last_n_blocks",
    "backbone",
    "pretrained_hf_id",
    "use_pretrained",
    "image_size",
    "batch_size",
    "tta_horizontal_flip",
    "imagenet_prior_blend",
    "imagenet_prior_model",
}
PREFERRED_COMPONENTS = {
    "finetune": 0,
    "resolution": 1,
    "optimizer": 2,
    "loss": 3,
    "scheduler": 4,
    "inference": 5,
    "augmentation": 6,
    "backbone": 7,
}


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_proposals(
    baseline: dict[str, Any],
    cards: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
    *,
    max_experiments: int = 3,
    max_changed_variables: int = 2,
) -> list[ExperimentProposal]:
    history = history or []
    rejected_hashes = {
        item.get("config_hash") or config_hash(item.get("config", {}))
        for item in history
        if item.get("status") in {"failed", "success"}
    }
    baseline_hash = config_hash(baseline)
    proposals: list[ExperimentProposal] = []
    cards = sorted(
        cards,
        key=lambda item: (
            PREFERRED_COMPONENTS.get(item.get("component", ""), 99),
            -float(item.get("priority", 0.0)),
        ),
    )
    for card in cards:
        template = dict(card.get("experiment_template") or {})
        unsupported = set(template) - SUPPORTED_EXPERIMENT_FIELDS
        if unsupported or not template:
            continue
        changed = [key for key, value in template.items() if baseline.get(key) != value]
        if not changed or len(changed) > max_changed_variables:
            continue
        if template.get("mixup_alpha", 0) and template.get("cutmix_alpha", 0):
            continue
        config = dict(baseline)
        config.update(template)
        digest = config_hash(config)
        if digest == baseline_hash or digest in rejected_hashes:
            continue
        rejected_hashes.add(digest)
        proposals.append(
            ExperimentProposal(
                experiment_name=f"exp_{len(proposals) + 1}_{card.get('strategy_name', 'strategy')}",
                strategy_card_ids=[card["id"]],
                config=config,
                changed_fields=changed,
                config_hash=digest,
                rationale=card.get("summary", ""),
            )
        )
        if len(proposals) >= max(0, min(int(max_experiments), 3)):
            break
    return proposals
