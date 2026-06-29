from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Path Utilities
# =============================================================================

def check_file_exists(path: Path) -> None:
    """
    Raise FileNotFoundError if the file does not exist.
    """
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")


def ensure_parent_dir(path: Path) -> None:
    """
    Create parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CSV
# =============================================================================

def save_csv(data: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_parent_dir(path)

    data.to_csv(path, index=index)

    logger.info(f"Saved CSV: {path}")


def load_csv(path: Path) -> pd.DataFrame:
    check_file_exists(path)

    df = pd.read_csv(path, sep=';')

    logger.info(f"Loaded CSV: {path}")

    return df


# =============================================================================
# NumPy
# =============================================================================

def save_numpy(data: np.ndarray, path: Path) -> None:
    ensure_parent_dir(path)

    np.save(path, data)

    logger.info(f"Saved NumPy array: {path}")


def load_numpy(path: Path) -> np.ndarray:
    check_file_exists(path)

    array = np.load(path, allow_pickle=False)

    logger.info(f"Loaded NumPy array: {path}")

    return array


# =============================================================================
# JSON
# =============================================================================

def save_json(data: dict[str, Any], path: Path, indent: int = 4) -> None:
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Saved JSON: {path}")


def load_json(path: Path) -> dict[str, Any]:
    check_file_exists(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded JSON: {path}")

    return data


# =============================================================================
# YAML
# =============================================================================

def save_yaml(data: dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    logger.info(f"Saved YAML: {path}")


def load_yaml(path: Path) -> dict[str, Any]:
    check_file_exists(path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.info(f"Loaded YAML: {path}")

    return data


# =============================================================================
# Models / Generic Python Objects
# =============================================================================

def save_model(model: Any, path: Path) -> None:
    """
    Save a model or any Python object using joblib.
    """
    ensure_parent_dir(path)

    joblib.dump(model, path)

    logger.info(f"Saved model: {path}")


def load_model(path: Path) -> Any:
    """
    Load a model or any Python object using joblib.
    """
    check_file_exists(path)

    model = joblib.load(path)

    logger.info(f"Loaded model: {path}")

    return model