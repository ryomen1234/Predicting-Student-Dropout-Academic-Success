import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)

def test_health_endpoint():
    """Health endpoint should return application status."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "message": "Application is working fine.",
    }

def test_predict_file_success(monkeypatch):
    """Prediction endpoint should return predictions."""

    df = pd.DataFrame({
        "Age": [18],
        "Target": ["Graduate"]
    })

    monkeypatch.setattr(main.pd, "read_csv", lambda *args, **kwargs: df)
    monkeypatch.setattr(main, "clean_data", lambda df, config: df)

    monkeypatch.setattr(
        main.validator,
        "validate_csv_file",
        lambda df: df,
    )

    monkeypatch.setattr(
        main.model_service,
        "predict",
        lambda df: ["Graduate"],
    )

    response = client.post(
        "/predict/file",
        files={
            "file": (
                "students.csv",
                io.BytesIO(b"dummy"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "filename": "students.csv",
        "rows": 1,
        "predictions": ["Graduate"],
    }

def test_predict_file_validation_error(monkeypatch):
    """Validation errors should return HTTP 400."""

    monkeypatch.setattr(
        main.pd,
        "read_csv",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(
        main,
        "clean_data",
        lambda df, config: df,
    )

    def fake_validate(df):
        raise ValueError("Missing required columns")

    monkeypatch.setattr(
        main.validator,
        "validate_csv_file",
        fake_validate,
    )

    response = client.post(
        "/predict/file",
        files={
            "file": (
                "students.csv",
                io.BytesIO(b"dummy"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Missing required columns"

def test_predict_file_internal_server_error(monkeypatch):
    """Unexpected exceptions should return HTTP 500."""

    def fake_read_csv(*args, **kwargs):
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(main.pd, "read_csv", fake_read_csv)

    response = client.post(
        "/predict/file",
        files={
            "file": (
                "students.csv",
                io.BytesIO(b"dummy"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."