from pathlib import Path
from functools import lru_cache

from utils.load_config import load_config
from utils.logger import get_logger
from utils.io import load_model

config = load_config()
logger = get_logger(__name__)

class ArtifactLoader:

    @staticmethod
    def _load(path: Path, name: str):
        try:
            artifact = load_model(path)
            logger.info("%s loaded successfully.", name)
            return artifact
        except Exception:
            logger.exception("Failed to load %s.", name)
            raise

    @staticmethod
    @lru_cache(maxsize=1)
    def get_model():
        return ArtifactLoader._load(
            config.artifacts.model_path,
            "Model",
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def get_encoder():
        return ArtifactLoader._load(
            config.artifacts.encoder_path,
            "Encoder",
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def get_feature_names():
        return ArtifactLoader._load(
            config.artifacts.feature_names_path,
            "Feature names",
        )