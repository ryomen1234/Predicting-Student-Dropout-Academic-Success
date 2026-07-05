from sklearn.model_selection import train_test_split

from utils.logger import get_logger
from utils.load_config import load_config
import pandas as pd 
import numpy as np 

logger = get_logger(__name__)

def split_data(config, df):

    logger.info("data split start:")
    TEST_SIZE = config.data_split.test_size
    RANDOM_STATE = config.data_split.random_state
    target = config.preprocessing.target_col
    logger.debug(f"test size: {TEST_SIZE}, random_state: {RANDOM_STATE}")
    
    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Training and test data created")
    logger.debug(f"X_train, y_train: {X_train.shape, y_train.shape} | X_test, y_test: {X_test.shape, y_test.shape}")

    return X_train, X_test, y_train, y_test



    
