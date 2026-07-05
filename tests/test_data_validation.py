import pandas as pd
import pytest

from src.data.data_validation import DataValidation


class DummyPreprocessing:
    target_col = "Target"
    target = ["Dropout", "Graduate", "Enrolled"]


class DummyConfig:
    preprocessing = DummyPreprocessing()


def test_validate_returns_summary():
    validator = DataValidation(DummyConfig())

    df = pd.DataFrame({
        "age": [18, 19, 20],
        "Target": ["Graduate", "Dropout", "Enrolled"]
    })

    result = validator.validate(df)

    assert result["data_shape"] == (3, 2)
    assert result["missing_values"] == {}
    assert result["duplicate_values"] == 0

def test_validate_detects_missing_values():
    validator = DataValidation(DummyConfig())

    df = pd.DataFrame({
        "age": [18, None, 20],
        "Target": ["Graduate", "Dropout", "Enrolled"]
    })

    result = validator.validate(df)

    assert result["missing_values"] == {"age": 1}

def test_validate_detects_duplicate_rows():
    validator = DataValidation(DummyConfig())

    df = pd.DataFrame({
        "age": [18, 18],
        "Target": ["Graduate", "Graduate"]
    })

    result = validator.validate(df)

    assert result["duplicate_values"] == 1

def test_validate_raises_error_for_invalid_target():
    validator = DataValidation(DummyConfig())

    df = pd.DataFrame({
        "age": [18, 19],
        "Target": ["Graduate", "Unknown"]
    })

    with pytest.raises(ValueError):
        validator.validate(df)

