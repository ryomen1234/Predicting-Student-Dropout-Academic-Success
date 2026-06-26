from sklearn.model_selection import train_test_split

from utils.logger import get_logger
from utils.load_config import load_config
import pandas as pd 
import numpy as np 

logger = get_logger(__name__)

def split_data(config):
    DATA_PATH = config.data_paths.data_dir_path / "raw" / "student_academic_data" / "data.csv"
    TEST_SIZE = config.data_split.test_size
    RANDOM_STATE = config.data_split.random_state
    logger.info(f"test size: {TEST_SIZE}, random_state: {RANDOM_STATE}")

    df = pd.read_csv(DATA_PATH, sep=';')
    
    X = df.drop(columns=['Target'])
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    logger.debug(f"Training and test data created")
    logger.debug(f"X_train, y_train: {X_train.shape, y_train.shape} | X_test, y_test: {X_test.shape, y_test.shape}")

    return X_train, X_test, y_train, y_test



    
