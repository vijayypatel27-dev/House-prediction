import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ensure_parent, load_config
from src.logger import get_logger

LOGGER = get_logger(__name__)


class DataValidationError(ValueError):
    pass


class DataPreprocessor:
    def __init__(self, config_path: str = "params.yaml") -> None:
        self.config = load_config(config_path)
        self.data_config = self.config["data"]
        self.feature_config = self.config["features"]

    def run(self) -> tuple[Path, Path, Path]:
        raw_path = Path(self.data_config["raw_path"])
        df = pd.read_csv(raw_path)
        self._validate(df)
        df = self._engineer_features(df)

        train_df, test_df = train_test_split(
            df,
            test_size=self.data_config["test_size"],
            random_state=self.data_config["random_state"],
        )

        preprocessor = self._build_preprocessor()
        target = self.feature_config["target"]
        preprocessor.fit(train_df.drop(columns=[target]))

        train_path = ensure_parent(self.data_config["processed_train_path"])
        test_path = ensure_parent(self.data_config["processed_test_path"])
        preprocessor_path = ensure_parent(self.config["training"]["preprocessor_path"])

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        joblib.dump(preprocessor, preprocessor_path)

        LOGGER.info("Saved processed splits and preprocessor")
        return train_path, test_path, preprocessor_path

    def _validate(self, df: pd.DataFrame) -> None:
        required = set(self.feature_config["numeric"] + self.feature_config["categorical"] + [self.feature_config["target"]])
        missing = required.difference(df.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {sorted(missing)}")
        if df.empty:
            raise DataValidationError("Dataset is empty")
        if (df[self.feature_config["target"]] <= 0).any():
            raise DataValidationError("Target price must be positive")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
        df["area_per_room"] = df["area"] / df["total_rooms"].replace(0, 1)
        for column in ["total_rooms", "area_per_room"]:
            if column not in self.feature_config["numeric"]:
                self.feature_config["numeric"].append(column)
        return df

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.feature_config["numeric"]),
                ("cat", categorical_pipeline, self.feature_config["categorical"]),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    DataPreprocessor(args.config).run()


if __name__ == "__main__":
    main()
