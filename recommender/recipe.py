"""Recipe layer — recommend training hyperparameters.

v0 (this): rules — hard backbone constraints + established finetuning conventions.
v1 (reserved): an LLM proposer grounded by the outcome memory, behind the same
`recommend_recipe(..., use_llm=True)` interface, with the rule recipe as the guardrailed
floor (clamp LLM proposals to valid ranges, fall back to rules on failure).

Logic basis:
  1. hard constraints — backbone facts (e.g. DINOv2 patch-14 → image_size multiple of 14).
     Lookups, always correct.
  2. conventions — lr scale by pretrained/strategy/family, augmentation by data size,
     early stopping. Encoded rules, grounded in finetuning practice (timm / HF / papers).
  3. (v1) data-driven calibration — outcome memory + LLM proposal.

Only emits keys the generated training code actually consumes today (learning_rate,
backbone_lr_scale, image_size, augmentation, early_stopping_patience). weight_decay /
warmup are conventions worth adding once the train template consumes them (see TODO).
"""

from __future__ import annotations

_TRANSFORMER_FAMILIES = ("vit", "swin", "dino", "clip", "deit", "beit", "eva")
_PATCH14_FAMILIES = ("dino",)  # DINOv2 uses patch-14 → image_size must be a multiple of 14

# early-stopping patience by data size (more data → can wait longer for improvement)
_EARLY_STOP_PATIENCE = {"small": 3, "medium": 5, "large": 8}


def _is_transformer(backbone: str) -> bool:
    return any(tok in (backbone or "").lower() for tok in _TRANSFORMER_FAMILIES)


def _recommend_image_size(backbone: str, m2_report: dict | None) -> int:
    """Pretrained-friendly default (224), bumped to 384 for clearly high-res sources;
    rounded to a patch multiple for patch-based transformers (DINOv2)."""
    size = 224
    if m2_report:
        w = float(m2_report.get("avg_width", 0) or 0)
        h = float(m2_report.get("avg_height", 0) or 0)
        avg = (w + h) / 2 if (w or h) else 0
        if avg >= 448:  # only clearly high-res — 384 ~ triples compute
            size = 384
    backbone = (backbone or "").lower()
    if any(tok in backbone for tok in _PATCH14_FAMILIES):
        size = round(size / 14) * 14  # e.g. 224 -> 224, 384 -> 392
    return size


def _rule_recipe(
    backbone: str,
    finetune_strategy: str | None,
    data_size: str,
    m2_report: dict | None,
    task_type: str,
) -> dict:
    recipe: dict = {}
    frozen = finetune_strategy == "head_only"
    transformer = _is_transformer(backbone)

    # learning_rate is the HEAD lr; the backbone group is scaled by backbone_lr_scale.
    if frozen:
        recipe["learning_rate"] = 1.0e-3          # linear probe tolerates a higher lr
    elif transformer:
        recipe["learning_rate"] = 1.0e-3          # head lr
        recipe["backbone_lr_scale"] = 0.01        # backbone ~1e-5 (avoid forgetting)
    else:  # CNN full finetune
        recipe["learning_rate"] = 3.0e-4

    recipe["image_size"] = _recommend_image_size(backbone, m2_report)
    recipe["augmentation"] = "strong" if data_size == "small" else "basic"
    recipe["early_stopping_patience"] = _EARLY_STOP_PATIENCE.get(data_size, 5)
    # TODO(v1): weight_decay / warmup once the train template consumes them.
    return recipe


def _llm_recipe_proposal(*args, **kwargs) -> dict | None:
    """v1 hook: an LLM proposes the soft HPs, grounded by the outcome memory, validated
    against the rule recipe's ranges. Not implemented yet — returns None so callers fall
    back to the rule recipe."""
    return None


def recommend_recipe(
    backbone: str,
    finetune_strategy: str | None = None,
    data_size: str = "medium",
    m2_report: dict | None = None,
    task_type: str = "classification",
    use_llm: bool = False,
    memory=None,
) -> dict:
    """Recommend training hyperparameters for one (backbone, strategy, data) setup.

    Returns a dict of config keys the pipeline injects into the Module 4 model_config.
    With `use_llm=True`, an LLM proposal (v1) refines the rule recipe within guardrails;
    today that path is a no-op stub, so the rule recipe is always returned.
    """
    recipe = _rule_recipe(backbone, finetune_strategy, data_size, m2_report, task_type)

    if use_llm:
        proposal = _llm_recipe_proposal(
            backbone=backbone, finetune_strategy=finetune_strategy,
            data_size=data_size, m2_report=m2_report, task_type=task_type,
            memory=memory, rule_recipe=recipe,
        )
        if isinstance(proposal, dict):
            # guardrail: only accept keys the rule recipe already defines (known + ranged)
            for key, value in proposal.items():
                if key in recipe:
                    recipe[key] = value

    return recipe
