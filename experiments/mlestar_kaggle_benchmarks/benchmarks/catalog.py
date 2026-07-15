"""Metric-correct contracts for the ten historical Kaggle benchmarks."""

from __future__ import annotations

from types import MappingProxyType

from mlestar.contracts import FoldSpec, MetricSpec, SubmissionSpec, TaskSpec


def _task(
    key: str,
    competition: str,
    modality: str,
    metric: str,
    target_columns: tuple[str, ...],
    *,
    id_columns: tuple[str, ...],
    prediction_columns: tuple[str, ...] = (),
    prediction_from_sample: bool = False,
    fold: FoldSpec | None = None,
    description: str,
) -> TaskSpec:
    return TaskSpec(
        key=key,
        competition=competition,
        modality=modality,
        metric=MetricSpec(metric),
        fold=fold or FoldSpec(n_splits=5),
        submission=SubmissionSpec(
            id_columns=id_columns,
            prediction_columns=prediction_columns,
            prediction_from_sample=prediction_from_sample,
        ),
        target_columns=target_columns,
        description=description,
    )


BENCHMARKS = MappingProxyType(
    {
        "leaf_classification": _task(
            "leaf_classification",
            "leaf-classification",
            "tabular_multiclass",
            "multiclass_log_loss",
            ("species",),
            id_columns=("id",),
            prediction_from_sample=True,
            description="Leaf species classification from precomputed features.",
        ),
        "plant_pathology_2020": _task(
            "plant_pathology_2020",
            "plant-pathology-2020-fgvc7",
            "image_multilabel",
            "roc_auc",
            ("healthy", "multiple_diseases", "rust", "scab"),
            id_columns=("image_id",),
            prediction_columns=("healthy", "multiple_diseases", "rust", "scab"),
            description="Four-label apple-leaf disease classification scored by mean ROC-AUC.",
        ),
        "aptos_2019": _task(
            "aptos_2019",
            "aptos2019-blindness-detection",
            "image_ordinal",
            "qwk",
            ("diagnosis",),
            id_columns=("id_code",),
            prediction_columns=("diagnosis",),
            description="Ordinal diabetic-retinopathy grading.",
        ),
        "dog_breed": _task(
            "dog_breed",
            "dog-breed-identification",
            "image_multiclass",
            "multiclass_log_loss",
            ("breed",),
            id_columns=("id",),
            prediction_from_sample=True,
            description="Fine-grained dog-breed classification.",
        ),
        "aerial_cactus": _task(
            "aerial_cactus",
            "aerial-cactus-identification",
            "image_binary",
            "roc_auc",
            ("has_cactus",),
            id_columns=("id",),
            prediction_columns=("has_cactus",),
            description="Binary cactus detection from aerial image tiles.",
        ),
        "dogs_vs_cats": _task(
            "dogs_vs_cats",
            "dogs-vs-cats-redux-kernels-edition",
            "image_binary",
            "log_loss",
            ("label",),
            id_columns=("id",),
            prediction_columns=("label",),
            description="Dog-versus-cat probability prediction.",
        ),
        "histopathologic_cancer": _task(
            "histopathologic_cancer",
            "histopathologic-cancer-detection",
            "image_binary",
            "roc_auc",
            ("label",),
            id_columns=("id",),
            prediction_columns=("label",),
            description="Metastatic cancer detection in image patches.",
        ),
        "global_wheat": _task(
            "global_wheat",
            "global-wheat-detection",
            "object_detection",
            "detection_map",
            ("bbox",),
            id_columns=("image_id",),
            prediction_columns=("PredictionString",),
            fold=FoldSpec(
                n_splits=5,
                strategy="group_kfold",
                shuffle=False,
                group_column="source",
            ),
            description="Wheat-head localization scored by mean AP over IoU thresholds.",
        ),
        "ultrasound_nerve": _task(
            "ultrasound_nerve",
            "ultrasound-nerve-segmentation",
            "image_segmentation",
            "dice",
            ("mask",),
            id_columns=("img",),
            prediction_columns=("pixels",),
            description="Ultrasound nerve segmentation with column-major RLE output.",
        ),
        "denoising_dirty_documents": _task(
            "denoising_dirty_documents",
            "denoising-dirty-documents",
            "image_denoising",
            "rmse",
            ("clean_image",),
            id_columns=("id",),
            prediction_columns=("value",),
            description="Pixel-level document denoising.",
        ),
    }
)


def get_task(key: str) -> TaskSpec:
    """Return an immutable benchmark contract by its stable command-line key."""

    try:
        return BENCHMARKS[key]
    except KeyError as error:
        available = ", ".join(BENCHMARKS)
        raise KeyError(f"Unknown benchmark {key!r}; choose one of: {available}") from error


def all_tasks() -> tuple[TaskSpec, ...]:
    """Return catalog tasks in deterministic insertion order."""

    return tuple(BENCHMARKS.values())
