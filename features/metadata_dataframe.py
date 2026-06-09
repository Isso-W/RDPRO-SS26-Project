import os

import pandas as pd


class MetadataDataFrameBuilder:

    def __init__(
        self,
        workspace_dir: str
    ):
        # Initialize builder

        self._workspace_dir = workspace_dir

    def build(self) -> dict:
        # Main entry point

        return self._build_dataframe()

    def _build_dataframe(self) -> dict:
        # Load CSV, validate columns, print summary, persist as Parquet

        df = self._load_metadata()

        self._validate_columns(df)

        self._print_summary(df)

        dataframe_path = self._save_dataframe(df)

        return {

            "dataframe_rows":
                len(df),

            "dataframe_columns":
                len(df.columns),

            "dataframe_path":
                dataframe_path
        }

    def _load_metadata(self) -> pd.DataFrame:
        # Read metadata.csv from the workspace directory

        metadata_path = os.path.join(
            self._workspace_dir,
            "metadata.csv"
        )

        return pd.read_csv(metadata_path)

    def _validate_columns(
        self,
        df: pd.DataFrame
    ) -> None:
        # Raise ValueError if any required column is absent

        required_columns = [
            "image_path",
            "class_name",
            "label_id",
            "split",
            "original_width",
            "original_height",
            "original_aspect_ratio",
            "original_mode",
            "original_format"
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def _print_summary(
        self,
        df: pd.DataFrame
    ) -> None:
        # Print row count, column count, column names and missing value counts

        print(f"Rows:         {len(df)}")

        print(f"Columns:      {len(df.columns)}")

        print(f"Column names: {list(df.columns)}")

        print("Missing values:")

        print(df.isnull().sum())

    def _save_dataframe(
        self,
        df: pd.DataFrame
    ) -> str:
        # Save DataFrame to Parquet and return the file path

        os.makedirs(
            self._workspace_dir,
            exist_ok=True
        )
        
        dataframe_path = os.path.join(
            self._workspace_dir,
            "dataframe.parquet"
        )

        df.to_parquet(
            dataframe_path,
            index=False
            )

        return {
            "dataframe_path": dataframe_path
        }
