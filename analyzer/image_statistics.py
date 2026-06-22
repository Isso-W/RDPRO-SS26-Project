from collections import Counter

from PIL import Image


class ImageStatisticsAnalyzer:

    def analyze(self, dataset):

        report = {}

        report.update(self._dataset_statistics(dataset))
        report.update(self._image_metadata(dataset))

        return report

    def _get_label_split(self, dataset):
        if "train" in dataset:
            return dataset["train"]
        return dataset[next(iter(dataset.keys()))]

    def _detect_annotation_format(self, split):
        columns = split.column_names

        if len(split) == 0:
            return "none", {}

        sample = split[0]

        if "label" in columns:
            val = sample["label"]
            if isinstance(val, Image.Image):
                return "segmentation_mask", {"column": "label"}
            return "classification", {"column": "label"}

        if "objects" in columns:
            obj = sample["objects"]
            label_key = self._find_label_key_in_objects(obj)
            if label_key:
                return "detection", {"column": "objects", "label_key": label_key}
            return "detection_unknown", {"column": "objects"}

        for col in ("annotation", "mask", "segmentation_map"):
            if col in columns:
                val = sample[col]
                if isinstance(val, Image.Image):
                    return "segmentation_mask", {"column": col}

        for col in ("labels", "categories"):
            if col in columns:
                return "detection_flat", {"column": col}

        return "none", {}

    def _find_label_key_in_objects(self, obj):
        candidates = ("category", "categories", "label", "labels", "class_id")
        if isinstance(obj, dict):
            for key in candidates:
                if key in obj:
                    return key
        elif isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], dict):
            for key in candidates:
                if key in obj[0]:
                    return key
        return None

    def _extract_detection_labels(self, split, column, label_key):
        sample = split[0]
        obj = sample[column]
        is_dict_of_lists = isinstance(obj, dict)

        labels = []
        for sample in split:
            obj = sample[column]
            if is_dict_of_lists:
                vals = obj.get(label_key, [])
                if isinstance(vals, list):
                    labels.extend(vals)
                else:
                    labels.append(vals)
            else:
                for item in obj:
                    if isinstance(item, dict) and label_key in item:
                        labels.append(item[label_key])
        return labels

    def _extract_segmentation_classes(self, split, column, max_samples=200):
        import random
        indices = list(range(len(split)))
        if len(indices) > max_samples:
            indices = random.sample(indices, max_samples)

        all_classes = set()
        for i in indices:
            mask = split[i][column]
            if hasattr(mask, "get_flattened_data"):
                all_classes.update(set(mask.get_flattened_data()))
            else:
                all_classes.update(set(mask.getdata()))
        all_classes.discard(0)
        return all_classes

    def _dataset_statistics(self, dataset):

        split_sizes = {
            split_name: len(dataset[split_name])
            for split_name in dataset.keys()
        }

        total_images = sum(split_sizes.values())

        label_split = self._get_label_split(dataset)
        fmt, info = self._detect_annotation_format(label_split)

        num_classes = None
        class_distribution = {}

        if fmt == "classification":
            # Column access returns the label column directly WITHOUT decoding the
            # image column — iterating rows would decode every image just to read labels.
            labels = list(label_split[info["column"]])
            num_classes = len(set(labels))
            class_distribution = dict(Counter(labels))

        elif fmt == "detection":
            labels = self._extract_detection_labels(
                label_split, info["column"], info["label_key"]
            )
            num_classes = len(set(labels))
            class_distribution = dict(Counter(labels))

        elif fmt == "detection_flat":
            labels = []
            for val in label_split[info["column"]]:   # column access, no image decode
                if isinstance(val, list):
                    labels.extend(val)
                else:
                    labels.append(val)
            num_classes = len(set(labels))
            class_distribution = dict(Counter(labels))

        elif fmt == "segmentation_mask":
            unique_classes = self._extract_segmentation_classes(
                label_split, info["column"]
            )
            num_classes = len(unique_classes)

        return {
            "split_sizes": split_sizes,
            "total_images": total_images,
            "num_classes": num_classes,
            "class_distribution": class_distribution,
            "annotation_format": fmt,
        }
        
    def _image_metadata(self, dataset, sample_size=500):
        """Image width/height + colour mode + format, in a single sampled pass.

        Each image accessed via ``dataset[split][i]`` is fully decoded, so we do it
        exactly once and on a random sample (these distributions don't need every
        image). Replaces the old three full-decode passes.
        """
        import random

        refs = [
            (split_name, idx)
            for split_name in dataset.keys()
            for idx in range(len(dataset[split_name]))
        ]
        if not refs:
            return {
                "min_width": 0, "max_width": 0, "avg_width": 0,
                "min_height": 0, "max_height": 0, "avg_height": 0,
                "mode_distribution": {}, "format_distribution": {},
                "metadata_sample_size": 0,
            }

        if len(refs) > sample_size:
            refs = random.Random(0).sample(refs, sample_size)

        widths, heights, modes, formats = [], [], [], []
        for split_name, idx in refs:
            image = dataset[split_name][idx]["image"]
            widths.append(image.width)
            heights.append(image.height)
            modes.append(image.mode)
            formats.append(str(image.format))

        n = len(widths)
        return {
            "min_width": min(widths),
            "max_width": max(widths),
            "avg_width": sum(widths) / n,
            "min_height": min(heights),
            "max_height": max(heights),
            "avg_height": sum(heights) / n,
            "mode_distribution": dict(Counter(modes)),
            "format_distribution": dict(Counter(formats)),
            "metadata_sample_size": n,
        }