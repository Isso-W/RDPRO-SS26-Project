from shlex import split

from datasets import load_dataset  

class ImageLoader:
    
    

    def load_dataset_by_name(
        self,
        dataset_id
    ):

        dataset = load_dataset(
            dataset_id
        )
        
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
            "dataset_name": dataset_id,
            "dataset": dataset
        }