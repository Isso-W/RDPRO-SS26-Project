import os

from ingestion.image_loader import ImageLoader
from analyzer.image_metadata import ImageMetadataExtractor
from analyzer.image_statistics import ImageStatisticsAnalyzer
from analyzer.image_quality import ImageQualityAnalyzer
from analyzer.summary_generator import SummaryGenerator
from processors.image_standardizer import ImageStandardizer
from features.feature_extractor import ImageFeatureExtractor



DATASET_ID = "uoft-cs/cifar10"


# Load dataset
loader = ImageLoader()

loaded = loader.load_dataset_by_name(
    DATASET_ID
)

dataset_name = loaded["dataset_name"]

safe_name = dataset_name.replace("/", "_")

workspace_dir = os.path.join(
        "workspace",
        safe_name
    )

dataset = loaded["dataset"]

print(dataset)
print(dataset.keys())

report = {}


# Metadata extraction

metadata_extractor = ImageMetadataExtractor(
    output_dir=workspace_dir
)

metadata_report = metadata_extractor.extract(
    dataset
)

report.update(
    metadata_report
)


# Statistics analysis
statistics_analyzer = ImageStatisticsAnalyzer()

report.update(
    statistics_analyzer.analyze(dataset)
)



# Quality analysis

quality_analyzer = ImageQualityAnalyzer()

quality_report = quality_analyzer.analyze(
    dataset
)

report.update(
    quality_report
)



# Image standardization

standardizer = ImageStandardizer(
    output_dir=os.path.join(
        workspace_dir,
        "processed_dataset"
    ),
    target_size=(224, 224),
    resize_mode="letterbox_resize"
)

print(
    "Starting image standardization..."
)

standardization_report = standardizer.process(
    dataset
)

print(
    "Image standardization completed."
)

report.update(
    standardization_report
)

print(standardization_report)



# Feature extraction

print(
    "Starting feature extraction..."
)

extractor = ImageFeatureExtractor(

    dataset_dir=os.path.join(
        workspace_dir, 
        "processed_dataset"
        ),

    output_dir=os.path.join(
        workspace_dir, 
        "features"
        ),

    batch_size=64
)

feature_report = extractor.extract()

print(
    "Feature extraction completed."
)

report.update(
    feature_report
)



# Save summary report
generator = SummaryGenerator()

safe_name = dataset_name.replace("/", "_")

generator.save_report(
    report=report,
    dataset_name=dataset_name,
    filename=os.path.join(
        workspace_dir,
        "report.json"
    )
)