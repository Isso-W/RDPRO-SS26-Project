# Vision Data Pipeline

A modular image data engineering pipeline for dataset ingestion, quality assessment, metadata extraction, image standardization, deep feature extraction, and agent-ready feature generation.

---

## Overview

This project provides an end-to-end pipeline for processing image datasets from Hugging Face and transforming them into machine-learning-ready and agent-ready assets.

The pipeline supports:

* Dataset ingestion
* Metadata extraction
* Dataset statistics analysis
* Image quality assessment
* Image standardization
* Deep feature extraction using ResNet50
* Tabular feature engineering preparation
* Future AI Agent integration

---

## Pipeline Architecture

```text
Dataset
    │
    ▼
Metadata Extraction
    │
    ├── metadata.csv
    │
    ▼
Statistics Analysis
    │
    ▼
Quality Analysis
    │
    ▼
Image Standardization
    │
    ├── processed_dataset/
    │
    ▼
Deep Feature Extraction
    │
    ├── features.npy
    ├── labels.npy
    ├── image_paths.json
    └── class_mapping.json
    │
    ▼
Tabular Feature Engineering (Skrub)
    │
    ▼
Agent Integration (Future)
```

---

## Project Structure

```text
image_data_processor/

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
│   └── feature_extractor.py
│
├── workspace/
│
└── main.py
```

---

## Components

### Dataset Ingestion

Loads image datasets directly from Hugging Face.

Outputs:

* Dataset object
* Dataset metadata

---

### Metadata Extraction

Extracts image-level metadata before any transformation.

Generated fields:

* image_path
* class_name
* label_id
* split
* original_width
* original_height
* original_aspect_ratio
* original_mode
* original_format

Output:

```text
metadata.csv
```

---

### Statistics Analysis

Computes dataset-wide statistics:

* Number of images
* Number of classes
* Class distribution
* Width statistics
* Height statistics
* Dataset split sizes

---

### Quality Analysis

Performs quality checks:

* Corrupted image detection
* Color mode distribution
* Resolution outlier detection

---

### Image Standardization

Converts images into a consistent format.

Operations:

* RGB conversion
* Letterbox resize
* JPEG conversion

Output:

```text
processed_dataset/
```

---

### Deep Feature Extraction

Uses a pretrained ResNet50 encoder to generate visual embeddings.

Output dimension:

```text
2048
```

Generated files:

```text
features.npy
labels.npy
image_paths.json
class_mapping.json
```

---

## Generated Outputs

```text
workspace/
└── dataset_name/

    metadata.csv

    report.json

    processed_dataset/

    features/
        features.npy
        labels.npy
        image_paths.json
        class_mapping.json
```

---

## Future Work

* Skrub-based tabular feature engineering
* Feature store generation
* Similarity search
* Vector database integration
* AI Agent querying layer
* Multimodal retrieval workflows

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Pipeline

```bash
python main.py
```
