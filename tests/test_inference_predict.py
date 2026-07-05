import pandas as pd
import numpy as np
import pytest

import src.inference.predict as service
from src.inference.predict import Modelservice


class FakeModel:
    def predict(self, X):
        return [2, 0]


class FakeEncoder:
      def inverse_transform(self, labels):
        return np.array([
            "Graduate",
            "Dropout"
        ])


@pytest.fixture
def sample_dataframe():
    """Sample input dataframe."""

    return pd.DataFrame({
        "Age": [18, 19],
        "Grade": [15, 16]
    })


def test_model_service_loads_artifacts(monkeypatch):
    """Modelservice should load model and encoder."""

    monkeypatch.setattr(
        service.ArtifactLoader,
        "get_model",
        lambda: FakeModel()
    )

    monkeypatch.setattr(
        service.ArtifactLoader,
        "get_encoder",
        lambda: FakeEncoder()
    )

    model_service = Modelservice()

    assert isinstance(model_service.model, FakeModel)
    assert isinstance(model_service.le, FakeEncoder)


def test_predict_returns_decoded_labels(monkeypatch, sample_dataframe):
    """Predictions should be converted back to class labels."""

    monkeypatch.setattr(
        service.ArtifactLoader,
        "get_model",
        lambda: FakeModel()
    )

    monkeypatch.setattr(
        service.ArtifactLoader,
        "get_encoder",
        lambda: FakeEncoder()
    )

    model_service = Modelservice()

    predictions = model_service.predict(sample_dataframe)

    assert predictions == [
        "Graduate",
        "Dropout",
    ]


def test_predict_returns_list(monkeypatch, sample_dataframe):
    """Prediction output should be returned as a Python list."""

    monkeypatch.setattr(
        service.ArtifactLoader,
        "get_model",
        lambda: FakeModel()
    )

    monkeypatch.setattr(
        service.ArtifactLoader,
        "get_encoder",
        lambda: FakeEncoder()
    )

    model_service = Modelservice()

    predictions = model_service.predict(sample_dataframe)

    assert isinstance(predictions, list)