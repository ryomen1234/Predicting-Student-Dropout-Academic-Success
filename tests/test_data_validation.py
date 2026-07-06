import pandas as pd
import pytest

from src.data.data_validation import DataValidation


class DummyPreprocessing:
    target_col = "Target"
    target = ["Dropout", "Graduate", "Enrolled"]


class DummyConfig:
    preprocessing = DummyPreprocessing()


@pytest.fixture
def validator():
    """Return a DataValidation instance."""
    return DataValidation(DummyConfig())


def test_validate_returns_summary(validator):
    """Validation should return the correct summary."""

    df = pd.DataFrame({
        "age": [18, 19, 20],
        "Target": ["Graduate", "Dropout", "Enrolled"],
    })

    result = validator.validate(df)

    assert result["data_shape"] == (3, 2)
    assert result["missing_values"] == {}
    assert result["duplicate_values"] == 0


def test_validate_detects_missing_values(validator):
    """Validation should report missing values."""

    df = pd.DataFrame({
        "age": [18, None, 20],
        "Target": ["Graduate", "Dropout", "Enrolled"],
    })

    result = validator.validate(df)

    assert result["missing_values"] == {"age": 1}


def test_validate_detects_duplicate_rows(validator):
    """Validation should detect duplicate rows."""

    df = pd.DataFrame({
        "age": [18, 19, 20, 18],
        "Target": [
            "Graduate",
            "Dropout",
            "Enrolled",
            "Graduate",
        ],
    })

    result = validator.validate(df)

    assert result["duplicate_values"] == 1


def test_validate_raises_error_for_invalid_target(validator):
    """Unknown target labels should raise ValueError."""

    df = pd.DataFrame({
        "age": [18, 19],
        "Target": ["Graduate", "Unknown"],
    })

    with pytest.raises(ValueError):
        validator.validate(df)