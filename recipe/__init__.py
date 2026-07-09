"""recipe 层 — 超参推荐层。

graph 决定"选什么组件"，recipe 决定"选出来的怎么配置"（image_size / lr /
epochs / augmentation），每个值带 provenance。纯规则、无 LLM、可单测。
规格见仓库根 recipe_layer_plan.md。
"""

from recipe.layer import build_recipe  # noqa: F401
