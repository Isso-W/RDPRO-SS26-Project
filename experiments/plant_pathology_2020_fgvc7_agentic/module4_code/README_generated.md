# Generated Module 4 Project

        This folder was generated from Module 3 candidate configurations. The
        structured `model_config` fields drive the generated code; task text is
        kept only for context. The project runs local smoke checks and does not
        perform long training.

        ## Candidates

        - rank 1: classification, backbone=dinov3, loss=cross_entropy_loss, optimizer=adamw, finetune=partial last2 (partial_last2)
- rank 2: classification, backbone=dinov3, loss=cross_entropy_loss, optimizer=adamw, finetune=partial last4 (partial_last4)
- rank 3: classification, backbone=dinov2, loss=cross_entropy_loss, optimizer=adamw, finetune=head_only
- rank 4: classification, backbone=swin_transformer, loss=focal_loss, optimizer=adamw, finetune=full

        ## Config Contract

        - `configs.json` contains the normalized Module 4 configs consumed by
          the generated scripts.
        - `generation_info.json` records whether `model.py` came from a model
          provider or from the template fallback.
        - `model_config` remains the provenance record from Module 3.
        - If `model_config` and natural-language `tasks` disagree, generated
          code follows the structured config.

        ## Code Generation

        - model.py source: `template`
        - configured provider: `none`
        - set `M4_LLM_PROVIDER=qwen` to request Qwen generation for `model.py`.
        - set `M4_LLM_PROVIDER=none` for template-only generation.

        ## Files

        - `configs.json`: normalized candidate configs used by this project.
        - `generation_info.json`: records provider and fallback status.
        - `utils.py`: shared config parsing, seed, and task-type helpers.
        - `model_utils.py`: shared backbone loading and freeze helpers.
        - `smoke_data.py`: shared synthetic data helpers for local smoke runs.
        - `model.py`: task-compatible PyTorch models with `build_model(config)`.
          Uses TinyBackbone in smoke mode, real pretrained backbone otherwise.
        - `train.py`: training loop with HuggingFace, Kaggle CSV/image, and
          ImageFolder dataloaders, strong augmentation, class weighting,
          mixed precision, validation, early stopping, resumable training,
          and best/last checkpoint saving when `offline_smoke: false`.
        - `evaluate.py`: metrics by task type.
        - `infer.py`: `predict(weights_path=None, image=None, config=None)`.
        - `run.py`: single-configuration runner (smoke or real).
        - `run_experiments.py`: sweeps every Module 3 candidate.

        ## Usage

        Smoke check (fast, offline, CPU):
        ```bash
        python run.py --config configs.json
        python run_experiments.py --input configs.json
        ```

        Real training (set `offline_smoke: false` in configs.json first):
        ```bash
        python run.py --config configs.json --epochs 20
        python run.py --config configs.json --dataset uoft-cs/cifar10 --epochs 10
        python run_experiments.py --input configs.json --epochs 5
        ```

        ## Smoke vs Real Training

        Smoke runs (`offline_smoke: true`, the default) never download weights:
        backbones are randomly initialized so the checks stay fast and offline.
        The local smoke path verifies tensor shapes, loss computation, backward
        pass, optimizer step, evaluation output, inference output, and
        experiment sweep coverage.

        For real training, set `offline_smoke: false` and keep
        `use_pretrained: true` in the config.  What changes:
        - `model.py` loads the real backbone via `model_utils.load_backbone`
          (HuggingFace checkpoint → torchvision → TinyBackbone fallback)
        - `train.py` loads either the HuggingFace dataset specified by
          `dataset_id`, a local CSV dataset specified by `train_csv`, or an
          ImageFolder dataset specified by `image_dir`
        - Multi-epoch training with per-epoch logging
        - Checkpoints saved to `checkpoints/` after each epoch
        - Requires: `pip install transformers datasets Pillow`

        ## Current Limitations

        - Real dataloader supports classification and feature_extraction;
          detection / segmentation still use synthetic data.
        - Object detection and segmentation metrics are simplified,
          not benchmark scores.
        - Module 3 controls candidate scale; this project only executes the
          supplied configs.


