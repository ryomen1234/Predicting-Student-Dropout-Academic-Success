import pytest
import pandas as pd

from src.model.eval import eval


class DummyPreprocessing:
    target_columns = [
        "Dropout",
        "Enrolled",
        "Graduate",
    ]


class DummyModel:
    model_name = "CatBoost"


class DummyConfig:
    preprocessing = DummyPreprocessing()
    model = DummyModel()


class FakeModel:
    def predict(self, X):
        return [0, 1, 2]


class FakeRun:
    def __init__(self):
        self.logged_data = None

    def log(self, data):
        self.logged_data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


@pytest.fixture
def sample_data():
    X_test = pd.DataFrame({
        "f1": [1, 2, 3]
    })

    y_test = [0, 1, 2]

    return X_test, y_test

def test_eval_returns_result_and_report(monkeypatch, sample_data):
    """Evaluation should return model name, score, and report."""

    fake_run = FakeRun()

    monkeypatch.setattr(
        "src.model.eval.wandb.init",
        lambda **kwargs: fake_run
    )

    X_test, y_test = sample_data

    result, report = eval(
        FakeModel(),
        X_test,
        y_test,
        DummyConfig()
    )

    assert result["model_name"] == "CatBoost"
    assert result["score"] == 1.0
    assert isinstance(report, str)

def test_metrics_are_logged_to_wandb(monkeypatch, sample_data):
    """Evaluation metrics should be logged to W&B."""

    fake_run = FakeRun()

    monkeypatch.setattr(
        "src.model.eval.wandb.init",
        lambda **kwargs: fake_run
    )

    X_test, y_test = sample_data

    eval(
        FakeModel(),
        X_test,
        y_test,
        DummyConfig()
    )

    assert fake_run.logged_data == {
        "model_name": "CatBoost",
        "f1_macro": 1.0,
    }

def test_model_predict_is_called(monkeypatch, sample_data):
    """Model prediction should be performed during evaluation."""

    fake_run = FakeRun()

    monkeypatch.setattr(
        "src.model.eval.wandb.init",
        lambda **kwargs: fake_run
    )

    class PredictSpy:
        def __init__(self):
            self.called = False

        def predict(self, X):
            self.called = True
            return [0, 1, 2]

    model = PredictSpy()

    X_test, y_test = sample_data

    eval(
        model,
        X_test,
        y_test,
        DummyConfig()
    )

    assert model.called

