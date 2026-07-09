"""kb_mining — Kaggle 优胜方案挖掘 → KB 增强。

一次性挖掘管道：harvest → extract → aggregate → decide。
各阶段幂等、可独立重跑，阶段间只通过 kb_mining/data/ 下的文件通信。
详见仓库根的 kb_mining_plan（实现规格）。
"""
