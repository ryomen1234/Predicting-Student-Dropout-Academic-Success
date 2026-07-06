from src.pipelines import (
    data_pipeline,
    training_pipeline,
    evaluation_pipeline,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def run():
    logger.info("=" * 60)
    logger.info("MAIN PIPELINE STARTED")
    logger.info("=" * 60)

    try:
        data_pipeline.run()
        training_pipeline.run()
        evaluation_pipeline.run()

        logger.info("=" * 60)
        logger.info("MAIN PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Pipeline execution failed.")
        raise


if __name__ == "__main__":
    run()