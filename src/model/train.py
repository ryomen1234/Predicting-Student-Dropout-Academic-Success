import wandb 
from catboost import CatBoostClassifier

from utils.logger import get_logger

logger = get_logger(__name__)

def train(X_train, y_train, config):
    logger.info("training started")
    params = config.model.best_params 
    logger.debug(f"parameters: {params}")
    
    model = CatBoostClassifier(**params)
    logger.debug(f"model: {type(model)}")

    model.fit(X_train, y_train)

    logger.info("model training complete")

    return model

       





        
