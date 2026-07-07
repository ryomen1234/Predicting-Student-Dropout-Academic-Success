from src.model import eval, train 

from utils.io import (
    load_csv,
    load_numpy,
    save_model,
    save_json
)

from utils.logger import get_logger
from utils.load_config import load_config

logger = get_logger(__name__)

def run():
    logger.info("========== Training Pipeline Started ==========")
   
    # load config
    config = load_config()
    logger.info("config file loaded")

    # load Training data
    X_train = load_csv(path=config.processed_data.X_train_path)
    y_train = load_numpy(path=config.processed_data.y_train_path)

    # model training
    model = train.train_model(X_train=X_train,
                              y_train=y_train,
                              config=config)
    

    save_model(model=model, path=config.artifacts.model_path)
    save_model(X_train.columns.tolist(), path=config.artifacts.feature_names_path)

    logger.info("Training Pipeline Complete.")

if __name__ == "__main__":
    run()










