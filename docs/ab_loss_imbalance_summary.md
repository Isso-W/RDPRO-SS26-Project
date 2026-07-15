# A/B arbitration summary: CE vs focal loss under `class_imbalance`

*2026-07-07. See `docs/ab_loss_imbalance_protocol.md` for the specification and `experiments/ab_loss_imbalance/` for the implementation.*

## Question

`kb_mining` found one conflict. For classification with `class_imbalance=True`, should the default loss remain **focal loss** (the current KB edge `focal_loss→cross_entropy_loss` plus the hard-coded `_select_components` rule), or should it become **cross-entropy** (the mined Kaggle consensus: support 0.71, dominance ×2.66, breadth 10)?

## Method (pre-registered)

- **Paired 5-fold evaluation**: Both arms, focal loss and cross-entropy, use the same stratified folds. Everything except the loss is fixed: EfficientNet-B0 at 224 px, 5 epochs, AdamW, ordinary shuffle sampling, and seed 42.
- **Decision rule**: For each fold, compute `Δ = metric(CE) − metric(focal)`. The tie band is `max(0.005, 2·SE)`. A result must clear the estimated noise to reverse the current default.
- **Planned testbeds**: `siim_isic` covers extreme imbalance in medical binary classification; `cassava` covers moderate imbalance in agricultural multiclass classification. This run completed only Cassava because SIIM-ISIC requires an approximately 23 GB download, three hours of training, and additional Colab compute units.

## Cassava result (primary metric: macro-F1, 5 folds)

| fold | Δ(CE − focal) |
|---|---|
| 0 | +0.0145 |
| 1 | +0.0041 |
| 2 | +0.0031 |
| 3 | +0.0041 |
| 4 | +0.0117 |

**Δ̄ = +0.0075, SE = 0.0023, tie band = ±0.005 → CE_WINS.** All five fold differences favor cross-entropy.

The raw records are in `experiments/ab_loss_imbalance/results/outcomes.jsonl` (10 records: 2 arms × 5 folds). Every record has the same `fold_file_sha256`, which verifies that the two arms used the same folds.

## Conclusion and KB disposition

- **Cassava**: Cross-entropy consistently outperformed focal loss. The mean difference cleared the tie band and agrees with the mined consensus, giving two independent sources of evidence.
- **Scope**: This result applies to moderately imbalanced, multiclass natural-image classification. It does not cover extreme imbalance or medical imaging, where focal loss may be more useful.
- **KB action**: One testbed and a narrow margin are not enough to reverse the global `focal→CE` edge. Mark the conflict **resolved-lean-CE**, retain focal loss as the default for now, and reconsider the edge after running the extreme-imbalance testbed.

## Reproduction

```bash
python -m pytest experiments/ab_loss_imbalance/tests -q
python -m experiments.ab_loss_imbalance.run_ab --testbed cassava
python -m experiments.ab_loss_imbalance.collect
```
