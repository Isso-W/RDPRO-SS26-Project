import os

from features.metadata_dataframe import MetadataDataFrameBuilder
from etl.skrub_processor import SkrubProcessor


DATASET_NAME = "uoft-cs/cifar10"


def main():

    safe_name = (
        DATASET_NAME.replace("/", "_")
    )

    workspace_dir = os.path.join(
        "workspace",
        safe_name
    )

    print(
        f"Using workspace: {workspace_dir}"
    )

    dataframe_builder = (
        MetadataDataFrameBuilder(
            workspace_dir=workspace_dir
        )
    )

    report = dataframe_builder.build()

    print(
        "\nDataFrame generation completed."
    )

    print(report)


    # Skrub feature engineering
    # Loads dataframe.parquet → saves engineered_features.parquet

    print(
        "\nStarting Skrub feature engineering..."
    )

    skrub_processor = SkrubProcessor(
        workspace_dir=workspace_dir
    )

    skrub_report = skrub_processor.process()

    report.update(skrub_report)

    print(
        "\nSkrub feature engineering completed."
    )

    print(skrub_report)


if __name__ == "__main__":

    main()