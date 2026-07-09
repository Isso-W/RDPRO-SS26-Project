# Plant Pathology 2020 FGVC7 Colab Runbook

This folder was generated locally with Jiaozi plus the agentic Kaggle workflow.
It is intended to be pushed to GitHub, cloned on Colab, and trained with a GPU.

## Competition Contract

- Kaggle slug: `plant-pathology-2020-fgvc7`
- Submission mode: classic `submission.csv`
- Target columns: `healthy`, `multiple_diseases`, `rust`, `scab`
- Local validation: stratified folds over the materialized single-label column
- Metric used in generated code: `roc_auc`, matching Kaggle's mean per-column AUC intent

## Colab Commands

Clone your branch:

```bash
git clone -b codex/rag-wang-fullchain-agentic https://github.com/Isso-W/Jiaozi.git
cd Jiaozi
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install kaggle transformers datasets timm scikit-learn
```

Provide Kaggle credentials in Colab, then accept the competition rules once in
the Kaggle UI. The helper reads the normal Kaggle locations or env vars.

Prepare data, one-hot labels, folds, and patched configs:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py
```

Train the first generated config, DINOv3-B partial last2:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py --train --epochs 5
```

Train every generated candidate with the same fold:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py --train-all --epochs 5
```

After training, write a submission receipt without submitting:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py --make-submission
```

Submit and log score when Kaggle exposes a public score:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py --make-submission --submit --log-memory
```

## Generated Candidates

- DINOv3-B/16, partial finetune last 2 blocks
- DINOv3-B/16, partial finetune last 4 blocks
- DINOv2-Base, head-only finetune
- Swin-Base, full finetune with focal loss

The helper enables class weights for all configs because `multiple_diseases` is
rare. Keep the same `folds.json` when comparing candidates.
