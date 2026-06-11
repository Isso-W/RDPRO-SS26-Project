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
                    f"数据集 {dataset_id!r} 包含多个子配置，必须指定 --subset 参数。\n"
                    f"原始错误: {msg}"
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