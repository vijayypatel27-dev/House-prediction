import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor, DataValidationError


def test_validation_rejects_missing_required_columns():
    preprocessor = DataPreprocessor("params.yaml")
    df = pd.DataFrame({"area": [1000], "price": [100000]})

    with pytest.raises(DataValidationError):
        preprocessor._validate(df)


def test_feature_engineering_adds_room_features():
    preprocessor = DataPreprocessor("params.yaml")
    df = pd.DataFrame(
        {
            "area": [1000],
            "bedrooms": [2],
            "bathrooms": [1],
        }
    )

    result = preprocessor._engineer_features(df)

    assert result.loc[0, "total_rooms"] == 3
    assert result.loc[0, "area_per_room"] == pytest.approx(333.333, rel=1e-3)
