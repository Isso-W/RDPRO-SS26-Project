"""layer.py — recipe 编排：(已选骨干 + 数据信号) → image_size/lr/epochs/augmentation/early stop。

每个值带 provenance（来源规则路径 + 信号缺失标记）。纯规则、无 LLM。
调用点：rag_retrieval.build_task_list 组装完 model_config 后。
"""

from __future__ import annotations

from recipe import augment, tables


def _resolve_image_size(backbone, backbone_facts, input_json, data_stats):
    """§3.2：checkpoint/family 基准 → fine_grained+高分辨率上调 → 整除吸附。"""
    notes = []
    facts = backbone_facts or {}
    base = facts.get("image_size") or facts.get("input_resolution")
    if base:
        notes.append(f"ckpt_default={base}")
    else:
        base = tables.family_image_default(backbone)
        notes.append(f"family_default={base}")
    size = int(base)

    # 上调：高分辨率数据 + 细粒度（需细节），但 speed/large 不上调
    c = input_json.get("constraints", {})
    res_tier = (data_stats or {}).get("resolution_tier")
    if (res_tier == "high" and c.get("fine_grained")
            and input_json.get("priority") != "speed"
            and input_json.get("data_size") != "large"):
        bumped = tables.bump_image(size)
        if bumped != size:
            notes.append(f"fine_grained+high_res→{bumped}")
            size = bumped
    elif res_tier is None:
        notes.append("res_signal_missing")

    # 硬约束吸附（安全，最后执行、不可跳过）
    divisor = tables.image_divisor(backbone)
    if divisor:
        snapped = tables.snap_to_divisor(size, divisor)
        if snapped != size:
            notes.append(f"snap/{divisor}→{snapped}")
            size = snapped
        else:
            notes.append(f"ok/{divisor}")

    return size, " | ".join(notes)


def build_recipe(
    config: dict,
    input_json: dict,
    backbone_facts: dict | None,
    data_stats: dict | None = None,
) -> tuple[dict, dict]:
    """→ (recipe, provenance)。recipe 并进 model_config，provenance 每字段一句来源。"""
    backbone = config.get("backbone")
    use_pretrained = bool(config.get("pretrained_hf_id"))
    strategy = config.get("finetune_strategy")
    data_size = input_json.get("data_size", "medium")
    mode = tables.training_mode(strategy, use_pretrained)

    recipe: dict = {}
    prov: dict = {}

    # 1. epochs（收编孤儿）
    recipe["epochs"] = tables.derive_recommended_epochs(data_size, strategy, use_pretrained)
    prov["epochs"] = f"epochs_table[{data_size},{mode}]"

    # 2. image_size
    recipe["image_size"], prov["image_size"] = _resolve_image_size(
        backbone, backbone_facts, input_json, data_stats)

    # 3. learning_rate（family_class × mode）
    fam_class = tables.family_class(backbone)
    recipe["learning_rate"] = tables.lr_base(fam_class, mode)
    prov["learning_rate"] = f"lr_base[{fam_class},{mode}]"

    # 4. augmentation（三维）
    recipe["augmentation"], prov["augmentation"] = augment.build_augment(
        config, input_json, data_stats)

    # 5. early stopping（只作为真实训练的保险丝，不改变 epoch 上限）
    recipe["early_stopping_patience"] = tables.early_stopping_patience(data_size)
    prov["early_stopping_patience"] = f"early_stopping[{data_size}]"

    return recipe, prov
