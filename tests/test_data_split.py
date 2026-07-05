import pandas as pd
import pytest

from src.data.split import split_data


class DummyDataSplit:
    test_size = 0.2
    random_state = 42


class DummyPreprocessing:
    target_col = "Target"


class DummyConfig:
    data_split = DummyDataSplit()
    preprocessing = DummyPreprocessing()


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for train-test splitting."""

    return pd.DataFrame({
        "Age": range(10),
        "Grade": range(10, 20),
        "Target": [
            "Graduate",
            "Dropout",
            "Graduate",
            "Enrolled",
            "Graduate",
            "Dropout",
            "Graduate",
            "Enrolled",
            "Graduate",
            "Dropout",
        ]
    })


def test_split_returns_expected_shapes(sample_dataframe):
    """Train and test datasets should have the expected sizes."""

    X_train, X_test, y_train, y_test = split_data(
        DummyConfig(),
        sample_dataframe
    )

    assert len(X_train) == 8
    assert len(X_test) == 2

    assert len(y_train) == 8
    assert len(y_test) == 2


def test_target_column_removed_from_features(sample_dataframe):
    """Feature datasets should not contain the target column."""

    X_train, X_test, _, _ = split_data(
        DummyConfig(),
        sample_dataframe
    )

    assert "Target" not in X_train.columns
    assert "Target" not in X_test.columns


def test_split_is_reproducible(sample_dataframe):
    """Using the same random state should produce identical splits."""

    first_split = split_data(DummyConfig(), sample_dataframe)
    second_split = split_data(DummyConfig(), sample_dataframe)

    pd.testing.assert_frame_equal(first_split[0], second_split[0])
    pd.testing.assert_frame_equal(first_split[1], second_split[1])
    pd.testing.assert_series_equal(first_split[2], second_split[2])
    pd.testing.assert_series_equal(first_split[3], second_split[3])