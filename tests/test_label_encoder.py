import pandas as pd
import pytest

from src.data.label_encoder import encode_label


@pytest.fixture
def sample_labels():
    """Sample train and test labels."""

    y_train = pd.Series([
        "Graduate",
        "Dropout",
        "Enrolled",
        "Graduate"
    ])

    y_test = pd.Series([
        "Dropout",
        "Graduate"
    ])

    return y_train, y_test


def test_encode_label_returns_encoded_labels_and_encoder(sample_labels):
    """Encoded labels should match the learned class mapping."""

    y_train, y_test = sample_labels

    y_train_trf, y_test_trf, encoder = encode_label(
        y_train,
        y_test
    )

    assert list(encoder.classes_) == [
        "Dropout",
        "Enrolled",
        "Graduate"
    ]

    assert list(y_train_trf) == [2, 0, 1, 2]
    assert list(y_test_trf) == [0, 2]


def test_encoded_output_has_same_length(sample_labels):
    """Encoding should preserve the number of samples."""

    y_train, y_test = sample_labels

    y_train_trf, y_test_trf, _ = encode_label(
        y_train,
        y_test
    )

    assert len(y_train_trf) == len(y_train)
    assert len(y_test_trf) == len(y_test)


def test_unknown_label_in_test_data_raises_error():
    """Unknown labels in the test set should raise ValueError."""

    y_train = pd.Series([
        "Graduate",
        "Dropout"
    ])

    y_test = pd.Series([
        "Unknown"
    ])

    with pytest.raises(ValueError):
        encode_label(y_train, y_test)