import pandas as pd
import pytest

import src.inference.validate as validation
from src.inference.validate import ValidateStudentData


@pytest.fixture
def expected_features():
    return [
        "Age",
        "Gender",
        "AdmissionGrade",
    ]


@pytest.fixture
def validator(monkeypatch, expected_features):
    """Validator with mocked feature names."""

    monkeypatch.setattr(
        validation.ArtifactLoader,
        "get_feature_names",
        lambda: expected_features,
    )

    return ValidateStudentData()


@pytest.fixture
def valid_dataframe():
    """Valid student dataframe."""

    return pd.DataFrame({
        "Age": [18, 19],
        "Gender": ["M", "F"],
        "AdmissionGrade": [150, 170],
    })


def test_validate_returns_dataframe(validator, valid_dataframe):
    """Valid input should return a DataFrame."""

    result = validator.validate_csv_file(valid_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_validate_reorders_columns(validator):
    """Columns should be reordered to match training order."""

    df = pd.DataFrame({
        "AdmissionGrade": [150],
        "Age": [18],
        "Gender": ["M"],
    })

    result = validator.validate_csv_file(df)

    assert list(result.columns) == [
        "Age",
        "Gender",
        "AdmissionGrade",
    ]


def test_validate_raises_for_missing_columns(validator):
    """Missing required columns should raise ValueError."""

    df = pd.DataFrame({
        "Age": [18],
        "Gender": ["M"],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        validator.validate_csv_file(df)


def test_validate_raises_for_extra_columns(validator):
    """Unexpected columns should raise ValueError."""

    df = pd.DataFrame({
        "Age": [18],
        "Gender": ["M"],
        "AdmissionGrade": [150],
        "Height": [170],
    })

    with pytest.raises(ValueError, match="Unexpected columns"):
        validator.validate_csv_file(df)


def test_validate_raises_for_invalid_input_type(validator):
    """Non-DataFrame input should raise TypeError."""

    with pytest.raises(TypeError):
        validator.validate_csv_file(["not", "a", "dataframe"])