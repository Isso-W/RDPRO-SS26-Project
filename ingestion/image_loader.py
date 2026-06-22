from datasets import load_dataset


class ImageLoader:

    def load_dataset_by_name(
        self,
        dataset_id,
        subset=None,
    ):

        try:
            dataset = load_dataset(dataset_id, subset)
        except ValueError as e:
            msg = str(e)
            if "Config name is missing" in msg or "pick one among" in msg.lower():
                raise ValueError(
                    f"Dataset {dataset_id!r} contains multiple configs; "
                    f"please specify --subset.\nOriginal error: {msg}"
                ) from e
            raise

        for split in dataset.keys():

            columns = dataset[split].column_names

            if (
                "img" in columns
                and
                "image" not in columns
            ):

                dataset[split] = (
                    dataset[split]
                    .rename_column(
                        "img",
                        "image"
                    )
                )

        return {
            "dataset_name": f"{dataset_id}/{subset}" if subset else dataset_id,
            "dataset": dataset
        }