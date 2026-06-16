import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import ensure_parent, load_config
from src.logger import get_logger
from src.model_evaluation import regression_metrics

LOGGER = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = "params.yaml") -> None:
        self.config = load_config(config_path)
        self.target = self.config["features"]["target"]
        self.training_config = self.config["training"]

    def run(self) -> dict[str, object]:
        train_df = pd.read_csv(self.config["data"]["processed_train_path"])
        test_df = pd.read_csv(self.config["data"]["processed_test_path"])
        preprocessor = joblib.load(self.training_config["preprocessor_path"])

        x_train, y_train = train_df.drop(columns=[self.target]), train_df[self.target]
        x_test, y_test = test_df.drop(columns=[self.target]), test_df[self.target]

        mlflow.set_experiment(self.training_config["experiment_name"])
        best = {"name": None, "model": None, "metrics": {"rmse": float("inf")}}

        for name, estimator, param_grid in self._models():
            with mlflow.start_run(run_name=name):
                pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
                model = self._fit_model(name, pipeline, param_grid, x_train, y_train)
                metrics = regression_metrics(y_test, model.predict(x_test))

                mlflow.log_params(self._flat_params(model))
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")
                LOGGER.info("%s metrics: %s", name, metrics)

                model_path = ensure_parent(Path(self.training_config["model_dir"]) / f"{name}.joblib")
                joblib.dump(model, model_path)

                if metrics["rmse"] < best["metrics"]["rmse"]:
                    best = {"name": name, "model": model, "metrics": metrics}

        best_model_path = ensure_parent(self.training_config["best_model_path"])
        metrics_path = ensure_parent(self.training_config["metrics_path"])
        joblib.dump(best["model"], best_model_path)
        payload = {"best_model": best["name"], **best["metrics"]}
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Best model: %s", best["name"])
        return payload

    def _models(self):
        models = self.training_config["models"]
        if models["linear_regression"]["enabled"]:
            yield "linear_regression", LinearRegression(), {}
        if models["random_forest"]["enabled"]:
            yield (
                "random_forest",
                RandomForestRegressor(random_state=self.training_config["random_state"]),
                {f"model__{k}": v for k, v in models["random_forest"]["param_grid"].items()},
            )
        if models["xgboost"]["enabled"]:
            try:
                from xgboost import XGBRegressor
            except ImportError:
                LOGGER.warning("xgboost is not installed; skipping XGBoostRegressor")
                return
            yield (
                "xgboost",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=self.training_config["random_state"],
                    n_jobs=1,
                ),
                {f"model__{k}": v for k, v in models["xgboost"]["param_grid"].items()},
            )

    def _fit_model(self, name, pipeline, param_grid, x_train, y_train):
        if not param_grid:
            pipeline.fit(x_train, y_train)
            return pipeline

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self.training_config["scoring"],
            cv=self.training_config["cv_folds"],
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        LOGGER.info("%s best params: %s", name, search.best_params_)
        return search.best_estimator_

    @staticmethod
    def _flat_params(model) -> dict[str, object]:
        return {
            key: value
            for key, value in model.get_params().items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    ModelTrainer(args.config).run()


if __name__ == "__main__":
    main()
