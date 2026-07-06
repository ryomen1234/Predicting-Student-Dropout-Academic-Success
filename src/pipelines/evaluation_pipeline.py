from src.model import eval

from utils.io import (
    load_model,
    load_csv,
    load_numpy,
    save_json,
    save_model
)

from utils.logger import get_logger
from utils.load_config import load_config

logger = get_logger(__name__)

def run():
    logger.info("=========== Evaluation Started =============")
   
    # load config
    config = load_config()
    logger.info("config file loaded")
    logger.debug(config)

    # load test data
    X_test = load_csv(path=config.processed_data.X_test_path)
    y_test = load_numpy(path=config.processed_data.y_test_path)

    # load model
    model = load_model(path=config.artifacts.model_path)

    # model evaluation 
    result, report = eval.eval(
        model=model,
        X_test=X_test,
        y_test=y_test,
        config=config
    )

    save_json(data=result, path=config.artifacts.eval_result_path)
    save_model(model=report, path=config.artifacts.classification_report_path)

    logger.info("========= Evaluation Complete ============")

if __name__ == "__main__":
    run()










