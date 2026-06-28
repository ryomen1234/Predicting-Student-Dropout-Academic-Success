from utils.logger import get_logger

from typing import Tuple
import pandas as pd 
import numpy as np 

logger = get_logger(__name__)

def clean_data(X_train: pd.DataFrame, X_test: pd.DataFrame, config) -> Tuple:

    feature_to_drop = config.preprocessing.drop_columns
    logger.debug(f"feature to drop has loaded: {feature_to_drop}")

    # clean columns names
    X_train.columns = X_train.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    X_test.columns = X_test.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    X_train = X_train.drop(columns=feature_to_drop)
    X_test = X_test.drop(columns=feature_to_drop)

    logger.debug(f"columns droped from X_train and X_test")
    logger.debug(f"X_train shape, X_test shape: {X_train.shape, X_test.shape}")

    return X_train, X_test






