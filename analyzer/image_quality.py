from collections import Counter
from shlex import split

class ImageQualityAnalyzer:


    def analyze(self, dataset):

        report = {}

        report.update(
            self._corrupted_images(dataset)
        )

        report.update(
            self._color_mode_counts(dataset)
        )

        report.update(
            self._resolution_outliers(dataset)
        )

        return report

    def _corrupted_images(self, dataset):

        corrupted_count = 0

        corrupted_examples = []

        for split in dataset.keys():

            for i, sample in enumerate(
                dataset[split]
            ):
                try:

                    sample["image"].tobytes()

                except Exception:

                    corrupted_count += 1

                    if len(corrupted_examples) < 20:

                        corrupted_examples.append(i)

        return {

            "corrupted_count":
                corrupted_count,

            "corrupted_examples":
                corrupted_examples
        }

    def _color_mode_counts(self, dataset):

        mode_counter = Counter()

        for split in dataset.keys():

            for sample in dataset[split]:

                mode_counter[
                    sample["image"].mode
                ] += 1

        return {

            "rgb_count":
                mode_counter.get("RGB", 0),

            "rgba_count":
                mode_counter.get("RGBA", 0),

            "grayscale_count":
                mode_counter.get("L", 0),

            "all_modes":
                dict(mode_counter)
        }

    def _resolution_outliers(self, dataset):

        samples = []

        for split in dataset.keys():

            for index, sample in enumerate(
                dataset[split]
            ):

                image = sample["image"]

                area = image.width * image.height

                samples.append({

                    "split": split,

                    "index": index,

                    "width": image.width,

                    "height": image.height,

                    "area": area
                })

                areas = [

                    sample["area"]

                    for sample in samples
                ]

                sorted_areas = sorted(areas)

                n = len(sorted_areas)

                q1 = sorted_areas[n // 4]

                q3 = sorted_areas[(3 * n) // 4]

                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr

                upper_bound = q3 + 1.5 * iqr

                outliers = [

                    sample

                    for sample in samples

                    if (
                        sample["area"] < lower_bound
                        or
                        sample["area"] > upper_bound
                    )
                ]

        return {

            "resolution_outlier_count":
                len(outliers),

            "resolution_outlier_examples":
                outliers[:20]
        }