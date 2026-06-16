from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config


class HousePricePredictor:
    def __init__(self, config_path: str = "params.yaml") -> None:
        self.config = load_config(config_path)
        model_path = Path(self.config["training"]["best_model_path"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run training first."
            )
        self.model = joblib.load(model_path)

    def predict(self, features: dict[str, Any]) -> float:
        row = pd.DataFrame([features])
        row["total_rooms"] = row["bedrooms"].astype(float) + row["bathrooms"].astype(float)
        row["area_per_room"] = row["area"].astype(float) / row["total_rooms"].replace(0, 1)
        return float(self.model.predict(row)[0])
