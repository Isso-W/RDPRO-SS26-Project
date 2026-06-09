# Vision Data Engineering Pipeline

## Overview

Vision Data Engineering Pipeline is an automated image dataset processing framework designed for large-scale computer vision datasets.

The project provides:

* Dataset ingestion
* Metadata extraction
* Dataset statistics analysis
* Image quality analysis
* Image standardization
* CNN feature extraction
* Metadata feature engineering using Skrub

The output can be consumed by downstream Machine Learning systems, Data Engineering workflows, or AI Agents.

---

## Architecture

Pipeline consists of two independent stages.

### Stage 1 — Image Processing Pipeline

Raw Dataset

↓

Metadata Extraction

↓

Statistics Analysis

↓

Quality Analysis

↓

Image Standardization

↓

CNN Feature Extraction

↓

Workspace Outputs

---

### Stage 2 — Metadata Feature Engineering Pipeline

metadata.csv

↓

DataFrame Conversion

↓

Skrub TableVectorizer

↓

Engineered Feature Table

↓

Agent / ML Consumption

---

## Stage 1 — Image Processing Pipeline

Entry Point:

```bash
python image_processor.py
```

Modules:

### ImageLoader

Loads image datasets from Hugging Face.

### ImageMetadataExtractor

Extracts original image metadata.

Outputs:

* image_path
* class_name
* label_id
* split
* original_width
* original_height
* original_aspect_ratio
* original_mode
* original_format

### ImageStatisticsAnalyzer

Computes:

* dataset size
* class distribution
* image dimensions
* resolution statistics

### ImageQualityAnalyzer

Computes:

* corrupted images
* color mode distribution
* format distribution
* resolution outliers

### ImageStandardizer

Performs:

* RGB conversion
* JPEG conversion
* image resizing
* letterbox padding

Output:

224×224 JPEG images

### ImageFeatureExtractor

Uses:

* ResNet50
* PyTorch
* torchvision

Outputs:

2048-dimensional feature vectors

Saved files:

* features.npy
* labels.npy
* image_paths.json
* class_mapping.json

---

## Stage 2 — Metadata Feature Engineering Pipeline

Entry Point:

```bash
python feature_processor.py
```

Modules:

### MetadataDataFrameBuilder

Converts:

```text
metadata.csv
```

into:

```text
dataframe.parquet
```

### SkrubProcessor

Uses:

```python
from skrub import TableVectorizer
```

Transforms metadata into machine-learning-ready features.

Numerical Features:

* original_width
* original_height
* original_aspect_ratio

Categorical Features:

* class_name
* split
* original_mode
* original_format

Preserved Columns:

* label_id
* image_path

Outputs:

* engineered_features.parquet
* feature_names.csv
* vectorizer.pkl

---

## Project Structure

```text
project/

├── ingestion/
│   └── image_loader.py
│
├── analyzer/
│   ├── image_metadata.py
│   ├── image_statistics.py
│   ├── image_quality.py
│   └── summary_generator.py
│
├── processors/
│   └── image_standardizer.py
│
├── features/
│   ├── feature_extractor.py
│   └── metadata_dataframe.py
│
├── etl/
│   └── skrub_processor.py
│
├── image_processor.py
├── feature_processor.py
│
└── workspace/
```

---

## Workspace Structure

Example:

```text
workspace/

└── uoft-cs_cifar10/

    ├── metadata.csv

    ├── dataframe.parquet

    ├── engineered_features.parquet

    ├── feature_names.csv

    ├── vectorizer.pkl

    ├── report.json

    ├── processed_dataset/

    └── features/
```

---

## Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run image processing:

```bash
python image_processor.py
```

Run metadata feature engineering:

```bash
python feature_processor.py
```

---

## Generated Artifacts

| File                        | Description                |
| --------------------------- | -------------------------- |
| metadata.csv                | Original image metadata    |
| report.json                 | Dataset analysis report    |
| processed_dataset/          | Standardized images        |
| features.npy                | CNN feature vectors        |
| labels.npy                  | Image labels               |
| image_paths.json            | Image path mapping         |
| class_mapping.json          | Class-to-label mapping     |
| dataframe.parquet           | Metadata DataFrame         |
| engineered_features.parquet | ML-ready metadata features |
| feature_names.csv           | Generated feature names    |
| vectorizer.pkl              | Trained Skrub vectorizer   |

---

## Future Work

Planned extensions:

* Image augmentation pipeline
* Feature store integration
* AI Agent integration
* Natural language dataset querying
* Automated dataset profiling
* Multi-modal data processing
* Vector database support
