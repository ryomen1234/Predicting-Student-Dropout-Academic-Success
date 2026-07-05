from src.model import train
from src.model import eval

from utils.io import (
    load_csv,
    save_model,
    save_json,
    load_numpy,
    load_model
)
from utils.logger import get_logger
from utils.load_config import load_config

logger = get_logger(__name__)


def run():

    logger.info("========== MODEL TRAINING PIPELINE STARTED ==========")

    config = load_config()

    # Load processed data
    X_train = load_csv(config.processed_path.X_train_path)
    y_train = load_numpy(config.processed_path.y_train_path)

    # Train model
    model = train.train_model(
        X_train=X_train,
        y_train=y_train,
        config=config,
    )

    # Save Model
    save_model(
        model=model,
        path=config.artifacts.model_path
    )

    # Save feature names
    save_model(
        X_train.columns.tolist(),
        config.artifacts.feature_names_path,
    )

    logger.info("========== MODEL TRAINING PIPELINE COMPLETED ==========")

if __name__ == "__main__":
    run()