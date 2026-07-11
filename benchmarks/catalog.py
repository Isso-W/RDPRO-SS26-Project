"""The ten vision benchmark contracts shown in the project competition sheet."""

from __future__ import annotations

from .contracts import BenchmarkContract, FoldContract, SubmissionContract
from mlestar.contracts import MetricSpec


def _contract(
    key: str, competition: str, modality: str, metric: str, greater: bool, labels: tuple[str, ...],
    submission: SubmissionContract, folds: FoldContract, query: str,
) -> BenchmarkContract:
    return BenchmarkContract(key, competition, modality, MetricSpec(metric, greater), labels, submission, folds, query)


BENCHMARKS = {
    "plant_pathology_2020": _contract("plant_pathology_2020", "plant-pathology-2020-fgvc7", "image_classification", "roc_auc", True, ("healthy", "multiple_diseases", "rust", "scab"), SubmissionContract("csv", "image_id", ("healthy", "multiple_diseases", "rust", "scab")), FoldContract("stratified"), "Multi-label apple leaf disease classification."),
    "aptos_2019": _contract("aptos_2019", "aptos2019-blindness-detection", "image_classification", "qwk", True, ("0", "1", "2", "3", "4"), SubmissionContract("csv", "id_code", ("diagnosis",)), FoldContract("stratified"), "Ordinal diabetic-retinopathy grading from retinal images."),
    "dog_breed": _contract("dog_breed", "dog-breed-identification", "image_classification", "multiclass_log_loss", False, (), SubmissionContract("csv", "id", ("breed_probabilities",)), FoldContract("stratified"), "Fine-grained dog breed probability prediction."),
    "global_wheat": _contract("global_wheat", "global-wheat-detection", "object_detection", "map_iou", True, ("wheat",), SubmissionContract("detection_csv", "image_id", ("PredictionString",)), FoldContract("group", group_column="source"), "Wheat-head object detection robust to source-domain shift."),
    "ultrasound_nerve": _contract("ultrasound_nerve", "ultrasound-nerve-segmentation", "image_segmentation", "dice", True, ("nerve",), SubmissionContract("rle_csv", "img", ("pixels",), rle_order="column_major"), FoldContract("group", group_column="subject"), "Binary ultrasound nerve segmentation."),
    "leaf_classification": _contract("leaf_classification", "leaf-classification", "tabular", "log_loss", False, (), SubmissionContract("csv", "id", ("species",)), FoldContract("stratified"), "Leaf species classification from tabular shape features."),
    "aerial_cactus": _contract("aerial_cactus", "aerial-cactus-identification", "image_classification", "roc_auc", True, ("has_cactus",), SubmissionContract("csv", "id", ("has_cactus",)), FoldContract("stratified"), "Binary aerial cactus identification."),
    "dogs_vs_cats": _contract("dogs_vs_cats", "dogs-vs-cats-redux-kernels-edition", "image_classification", "log_loss", False, ("cat", "dog"), SubmissionContract("csv", "id", ("label",)), FoldContract("stratified"), "Binary dog-versus-cat image probability prediction."),
    "histopathologic_cancer": _contract("histopathologic_cancer", "histopathologic-cancer-detection", "image_classification", "roc_auc", True, ("tumor",), SubmissionContract("csv", "id", ("label",)), FoldContract("stratified"), "Binary histopathologic cancer patch classification."),
    "denoising_dirty_documents": _contract("denoising_dirty_documents", "denoising-dirty-documents", "image_to_image", "rmse", False, (), SubmissionContract("image_directory", "image"), FoldContract("kfold"), "Pixel-level document image denoising."),
}

ALIASES = {contract.competition: key for key, contract in BENCHMARKS.items()}


def get_benchmark(key: str) -> BenchmarkContract:
    canonical = ALIASES.get(key, key)
    try:
        return BENCHMARKS[canonical]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark {key!r}. Available: {', '.join(sorted(BENCHMARKS))}") from exc


def list_benchmarks() -> tuple[BenchmarkContract, ...]:
    return tuple(BENCHMARKS[key] for key in sorted(BENCHMARKS))
