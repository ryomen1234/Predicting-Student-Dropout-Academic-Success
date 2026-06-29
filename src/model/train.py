from time import perf_counter 
from catboost import CatBoostClassifier
from utils.logger import get_logger

logger = get_logger(__name__)

def train_model(X_train, y_train, config) -> CatBoostClassifier:
    logger.info("Starting CatBoost Training")
    logger.info(
        "Training data shape: %d sample, %d feature",
        X_train.shape[0],
        X_train.shape[1]
    )

    params = config.model.best_params 
    logger.debug(f"Hyperparameters: {params}")

    start = perf_counter()
    try:
      model = CatBoostClassifier(**params)
      model.fit(X_train, y_train)
    except Exception:
       logger.exception("Model training failed")
    
    elapsed = perf_counter() - start
    logger.info("Training completed in %.2f seconds", elapsed)

    return model

       





        
