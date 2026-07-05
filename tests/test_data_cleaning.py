import pandas as pd
import pytest

from src.data.data_cleaning import clean_data


class DummyPreprocessing:
    def __init__(self, drop_columns):
        self.drop_columns = drop_columns


class DummyConfig:
    def __init__(self, drop_columns):
        self.preprocessing = DummyPreprocessing(drop_columns)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        " Age ": [18, 19],
        " Name ": ["John", "Jane"],
        " Target ": ["Graduate", "Dropout"]
    })


def test_drop_configured_columns(sample_df):
    """Configured columns should be removed."""

    config = DummyConfig(["Age"])

    result = clean_data(sample_df, config)

    assert list(result.columns) == ["Name", "Target"]


def test_clean_column_names():
    """Column names should be stripped and consecutive spaces collapsed."""

    df = pd.DataFrame({
        " Previous    qualification ": [1],
        " Target ": ["Graduate"]
    })

    config = DummyConfig([])

    result = clean_data(df, config)

    assert list(result.columns) == [
        "Previous qualification",
        "Target",
    ]


def test_ignore_missing_columns():
    """Dropping a non-existent column should not raise an error."""

    df = pd.DataFrame({
        "Age": [18],
        "Target": ["Graduate"]
    })

    config = DummyConfig(["NotExist"])

    result = clean_data(df, config)

    assert list(result.columns) == ["Age", "Target"]