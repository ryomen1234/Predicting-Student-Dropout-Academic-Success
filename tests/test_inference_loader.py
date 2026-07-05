import pytest

import src.inference.loader as loader
from src.inference.loader import ArtifactLoader


def setup_function():
    """Clear lru_cache before each test."""

    ArtifactLoader.get_model.cache_clear()
    ArtifactLoader.get_encoder.cache_clear()
    ArtifactLoader.get_feature_names.cache_clear()


def test_load_returns_loaded_artifact(monkeypatch):
    """_load should return the loaded artifact."""

    def fake_load_model(path):
        return {"artifact": path}

    monkeypatch.setattr(loader, "load_model", fake_load_model)

    result = ArtifactLoader._load("model.pkl", "Model")

    assert result == {"artifact": "model.pkl"}


def test_load_raises_exception_when_loading_fails(monkeypatch):
    """_load should propagate exceptions from load_model."""

    def fake_load_model(path):
        raise FileNotFoundError("Artifact not found")

    monkeypatch.setattr(loader, "load_model", fake_load_model)

    with pytest.raises(FileNotFoundError):
        ArtifactLoader._load("model.pkl", "Model")


def test_get_model_uses_cache(monkeypatch):
    """Model should only be loaded once because of lru_cache."""

    calls = 0

    def fake_load_model(path):
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(loader, "load_model", fake_load_model)

    model_1 = ArtifactLoader.get_model()
    model_2 = ArtifactLoader.get_model()

    assert calls == 1
    assert model_1 is model_2