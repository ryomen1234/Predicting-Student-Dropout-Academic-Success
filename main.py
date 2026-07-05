from src.pipelines import data_pipeline
from src.pipelines import model_train_pipeline
from src.pipelines import model_eval_pipeline

from utils.logger import get_logger

logger = get_logger(__name__)


def run():

    logger.info("========== TRAINING PIPELINE STARTED ==========")

    data_pipeline.run()
    model_train_pipeline.run()
    model_eval_pipeline.run()

    logger.info("========== TRAINING PIPELINE COMPLETED ==========")


if __name__ == "__main__":
    run()