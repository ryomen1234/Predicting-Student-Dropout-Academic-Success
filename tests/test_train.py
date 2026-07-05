import numpy as np
import pandas as pd
import pytest

from src.model.train import train_model


class DummyModelConfig:
    best_params = {
        "iterations": 10,
        "verbose": False,
    }


class DummyConfig:
    model = DummyModelConfig()


@pytest.fixture
def sample_data():
    """Small dataset for model training."""

    X = pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [5, 6, 7, 8],
    })

    y = np.array([0, 1, 0, 1])

    return X, y


class FakeCatBoostClassifier:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.fit_called = False

    def fit(self, X, y):
        self.fit_called = True
        self.X = X
        self.y = y

def test_train_model_returns_fitted_model(monkeypatch, sample_data):
    """Model should be created and fitted."""

    monkeypatch.setattr(
        "src.model.train.CatBoostClassifier",
        FakeCatBoostClassifier
    )

    X, y = sample_data

    model = train_model(X, y, DummyConfig())

    assert isinstance(model, FakeCatBoostClassifier)
    assert model.fit_called

def test_model_receives_config_parameters(monkeypatch, sample_data):
    """Model should receive parameters from the config."""

    monkeypatch.setattr(
        "src.model.train.CatBoostClassifier",
        FakeCatBoostClassifier
    )

    X, y = sample_data

    model = train_model(X, y, DummyConfig())

    assert model.params == DummyConfig.model.best_params

def test_training_error_is_propagated(monkeypatch, sample_data):
    """Training failures should raise an exception."""

    class BrokenModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise RuntimeError("Training failed")

    monkeypatch.setattr(
        "src.model.train.CatBoostClassifier",
        BrokenModel
    )

    X, y = sample_data

    with pytest.raises(RuntimeError):
        train_model(X, y, DummyConfig())