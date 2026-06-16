import argparse
from pathlib import Path

import pandas as pd

from src.config import ensure_parent, load_config
from src.logger import get_logger

LOGGER = get_logger(__name__)


class DataIngestion:
    """Loads source data or creates a deterministic sample dataset."""

    def __init__(self, config_path: str = "params.yaml") -> None:
        self.config = load_config(config_path)
        self.raw_path = Path(self.config["data"]["raw_path"])

    def run(self) -> Path:
        ensure_parent(self.raw_path)
        if self.raw_path.exists():
            LOGGER.info("Raw dataset already exists at %s", self.raw_path)
            return self.raw_path

        LOGGER.info("Creating sample house price dataset at %s", self.raw_path)
        data = self._sample_data()
        data.to_csv(self.raw_path, index=False)
        return self.raw_path

    @staticmethod
    def _sample_data() -> pd.DataFrame:
        rows = [
            [7420, 4, 2, 3, 2, 5, "yes", "no", "no", "no", "yes", "yes", "furnished", "central", 13300000],
            [8960, 4, 4, 4, 3, 4, "yes", "no", "no", "no", "yes", "no", "furnished", "central", 12250000],
            [9960, 3, 2, 2, 2, 8, "yes", "no", "yes", "no", "no", "yes", "semi-furnished", "suburban", 12250000],
            [7500, 4, 2, 2, 3, 10, "yes", "no", "yes", "no", "yes", "yes", "furnished", "central", 12215000],
            [6550, 4, 2, 2, 1, 12, "yes", "yes", "yes", "no", "yes", "no", "semi-furnished", "central", 11410000],
            [3500, 3, 1, 1, 0, 18, "yes", "no", "no", "no", "no", "no", "unfurnished", "outer", 3430000],
            [7800, 3, 2, 2, 2, 7, "yes", "yes", "no", "no", "yes", "yes", "furnished", "central", 9870000],
            [6000, 3, 1, 2, 1, 14, "yes", "no", "yes", "no", "yes", "no", "semi-furnished", "suburban", 6293000],
            [4500, 3, 1, 2, 0, 16, "yes", "no", "no", "no", "no", "no", "unfurnished", "outer", 4095000],
            [4000, 2, 1, 1, 0, 20, "yes", "no", "no", "no", "no", "no", "unfurnished", "outer", 3150000],
            [8500, 4, 3, 3, 2, 6, "yes", "yes", "yes", "no", "yes", "yes", "furnished", "central", 10850000],
            [6600, 3, 2, 2, 1, 11, "yes", "no", "yes", "no", "yes", "yes", "semi-furnished", "suburban", 7210000],
            [5200, 3, 1, 1, 1, 22, "yes", "no", "no", "no", "no", "no", "semi-furnished", "outer", 4550000],
            [3000, 2, 1, 1, 0, 25, "no", "no", "no", "no", "no", "no", "unfurnished", "outer", 2400000],
            [10500, 5, 3, 4, 3, 3, "yes", "yes", "yes", "yes", "yes", "yes", "furnished", "central", 14000000],
            [7000, 3, 2, 2, 2, 9, "yes", "no", "yes", "no", "yes", "yes", "semi-furnished", "central", 8645000],
            [6100, 3, 1, 2, 1, 13, "yes", "yes", "no", "no", "yes", "no", "semi-furnished", "suburban", 5950000],
            [3600, 2, 1, 1, 0, 19, "yes", "no", "no", "no", "no", "no", "unfurnished", "outer", 2940000],
            [9000, 4, 3, 3, 2, 4, "yes", "yes", "yes", "no", "yes", "yes", "furnished", "central", 11900000],
            [4800, 3, 1, 2, 1, 17, "yes", "no", "no", "no", "no", "no", "semi-furnished", "outer", 4200000],
        ]
        columns = [
            "area", "bedrooms", "bathrooms", "stories", "parking", "age",
            "mainroad", "guestroom", "basement", "hotwaterheating",
            "airconditioning", "prefarea", "furnishingstatus", "city_zone", "price",
        ]
        return pd.DataFrame(rows, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    DataIngestion(args.config).run()


if __name__ == "__main__":
    main()
