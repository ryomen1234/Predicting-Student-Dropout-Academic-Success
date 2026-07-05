from utils.logger import get_logger

from typing import Tuple
import pandas as pd 
import numpy as np 

logger = get_logger(__name__)

def clean_data(data: pd.DataFrame, config) -> pd.DataFrame:
    logger.info("Data cleaning started")

    data = data.copy()
    
    feature_to_drop = config.preprocessing.drop_columns
    logger.debug(f"feature to drop has loaded: {feature_to_drop}")

    # clean columns names
    data.columns = data.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    data = data.drop(columns=feature_to_drop, errors="ignore")

    logger.info("columns droped from X_train and X_test")
    logger.debug(f"data shape: {data.shape}")

    logger.info("Data cleaning complete")

    return data






