from src.model import train
from src.model import eval

from utils.io import (
    load_csv,
    save_model,
    save_json,
    load_model,
    load_numpy
)
from utils.logger import get_logger
from utils.load_config import load_config

logger = get_logger(__name__)


def run():

    logger.info("========== MODEL PIPELINE STARTED ==========")

    config = load_config()

    # Load processed data
    X_test = load_csv(config.processed_path.X_test_path)
    y_test = load_numpy(config.processed_path.y_test_path)

    # Load Model
    model = load_model(
        path=config.artifacts.model_path
    )

    # Evaluate
    result, report = eval.eval(
        model=model,
        X_test=X_test,
        y_test=y_test,
        config=config,
    )

    # Save evaluation metrics
    save_json(
        result,
        config.artifacts.evaluation_path,
    )

    # Save classification report
    save_model(
        report,
        config.artifacts.classification_report_path,
    )

    logger.info("========== MODEL PIPELINE COMPLETED ==========")

if __name__ == "__main__":
    run()
    