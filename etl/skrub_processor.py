import os

import joblib
import pandas as pd

from skrub import TableVectorizer


class SkrubProcessor:
    # Engineer features from a metadata DataFrame using Skrub's TableVectorizer.
    #
    # Pipeline:
    #   workspace/<dataset>/dataframe.parquet
    #       -> TableVectorizer (numerical + categorical columns)
    #       -> engineered_features.parquet  (+ preserved columns)
    #       -> vectorizer.pkl
    #       -> feature_names.csv

    # Columns fed to the vectorizer as numerical features
    NUMERICAL_COLUMNS = [
        "original_width",
        "original_height",
        "original_aspect_ratio"
    ]

    # Columns fed to the vectorizer as categorical features
    CATEGORICAL_COLUMNS = [
        "class_name",
        "split",
        "original_mode",
        "original_format"
    ]

    # Columns carried through untouched and appended back after transformation
    PRESERVED_COLUMNS = [
        "label_id",
        "image_path"
    ]

    def __init__(
        self,
        workspace_dir: str
    ):
        # Initialize processor with the dataset workspace directory

        self._workspace_dir = workspace_dir

        # Resolve all output paths once, relative to the workspace directory

        self._dataframe_path = os.path.join(
            self._workspace_dir,
            "dataframe.parquet"
        )

        self._engineered_features_path = os.path.join(
            self._workspace_dir,
            "engineered_features.parquet"
        )

        self._vectorizer_path = os.path.join(
            self._workspace_dir,
            "vectorizer.pkl"
        )

        self._feature_names_path = os.path.join(
            self._workspace_dir,
            "feature_names.csv"
        )

    def process(self) -> dict:
        # Main entry point: load, transform, persist artifacts and return a report

        df = self._load_dataframe()

        engineered_df, vectorizer, feature_names = (
            self._transform_dataframe(df)
        )

        self._save_vectorizer(vectorizer)

        self._save_feature_names(feature_names)

        self._save_dataframe(engineered_df)

        return {

            "engineered_rows":
                len(engineered_df),

            "engineered_columns":
                len(engineered_df.columns),
                
            "engineered_feature_count":
                len(feature_names),

            "engineered_features_path":
                self._engineered_features_path,

            "vectorizer_path":
                self._vectorizer_path,

            "feature_names_path":
                self._feature_names_path
        }

    def _load_dataframe(self) -> pd.DataFrame:
        # Read dataframe.parquet from the workspace directory using pandas

        if not os.path.exists(self._dataframe_path):

            raise FileNotFoundError(
                f"DataFrame not found: {self._dataframe_path}"
            )

        df = pd.read_parquet(self._dataframe_path)

        print(f"Input shape:  {df.shape}")

        return df

    def _transform_dataframe(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, TableVectorizer, list[str]]:
        # Vectorize the feature columns and re-attach the preserved columns.
        # Returns the engineered DataFrame, the fitted vectorizer and the
        # list of engineered feature names.

        # Guard against schema drift before indexing into the DataFrame
        self._validate_columns(df)

        feature_columns = (
            self.NUMERICAL_COLUMNS
            + self.CATEGORICAL_COLUMNS
        )

        # Split the table: features go to the vectorizer, preserved
        # columns are kept aside and stitched back on afterwards.
        features = df[feature_columns]

        preserved = (
            df[self.PRESERVED_COLUMNS]
            .reset_index(drop=True)
        )

        vectorizer = TableVectorizer()

        # fit_transform returns a pandas DataFrame with named columns
        transformed = vectorizer.fit_transform(features)

        if isinstance(
            transformed,
            pd.DataFrame
        ):
            result_df = transformed

        else:
            result_df = pd.DataFrame(
                transformed,
                columns=vectorizer.get_feature_names_out()
            )

        result_df = result_df.reset_index(
            drop=True
        )

        # Engineered feature names exclude the preserved columns by construction
        feature_names = [
            str(column)
            for column in result_df.columns
        ]

        # Append the untouched columns back onto the engineered features
        engineered_df = pd.concat(
            [result_df, preserved],
            axis=1
        )

        print(f"Output shape: {engineered_df.shape}")

        print(
            f"Generated feature count: {len(feature_names)}"
        )

        print("Column dtypes:")

        print(engineered_df.dtypes)

        return engineered_df, vectorizer, feature_names

    def _validate_columns(
        self,
        df: pd.DataFrame
    ) -> None:
        # Raise ValueError if any required column is absent

        required_columns = (
            self.NUMERICAL_COLUMNS
            + self.CATEGORICAL_COLUMNS
            + self.PRESERVED_COLUMNS
        )

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def _save_dataframe(
        self,
        df: pd.DataFrame
    ) -> str:
        # Persist the engineered features to Parquet and return the path

        os.makedirs(
            self._workspace_dir,
            exist_ok=True
        )

        df.to_parquet(
            self._engineered_features_path,
            index=False
        )

        return self._engineered_features_path

    def _save_feature_names(
        self,
        feature_names: list[str]
    ) -> str:
        # Write the engineered feature names to a single-column CSV.
        # The preserved columns are intentionally excluded.

        os.makedirs(
            self._workspace_dir,
            exist_ok=True
        )

        feature_names_df = pd.DataFrame(
            {"feature_name": feature_names}
        )

        feature_names_df.to_csv(
            self._feature_names_path,
            index=False
        )

        return self._feature_names_path

    def _save_vectorizer(
        self,
        vectorizer: TableVectorizer
    ) -> str:
        # Serialize the fitted vectorizer with joblib for later reuse

        os.makedirs(
            self._workspace_dir,
            exist_ok=True
        )

        joblib.dump(
            vectorizer,
            self._vectorizer_path
        )

        return self._vectorizer_path
