
# data validation

# 1. missing value 
# 2. duplicate values
# 3. dataset shape 
# 4. paths exist or not 

import pandas as pd 
import numpy as np
from pathlib import Path 

from utils.logger import get_logger

logger = get_logger(__name__)

class DataValidation:
    '''
    This class validate data.
    it check:

    1. missing value 
    2. duplicate values
    3. dataset shape 
    4. paths exist or not 

    '''

    def __init__(self, config) -> None:
        self.config = config
        self.data_path = config.data_paths.data_dir_path / "raw" / "student_academic_data" / "data.csv"
        self.target = config.preprocessing.target_columns

    def check_paths(self):
        if not self.data_path.exists():
            logger.error(f"{self.data_path} not found.")
            raise FileNotFoundError(f"{self.data_path} not found.")
    
    def main(self):
        logger.info("Data validation started.")
        
        self.check_paths()

        df = pd.read_csv(self.data_path, sep=";")
        logger.info("Data loaded.")

        data_shape = df.shape
        missing_vals = df.isnull().sum()
        miss_val = len(missing_vals[missing_vals > 0])
        duplicated_value = df.duplicated().sum()

        logger.debug(f"data shape: {data_shape}")
        logger.debug(f"missing values: {miss_val}")
        logger.debug(f"duplicated values : {duplicated_value}")

        if set(df["Target"].unique()) != set(self.target):
            logger.error(f"target column {self.target} mismatch")
            raise ValueError("target column don't match: {self.target}")
        
        result = {
            "data_shape": data_shape,
            "missing_values": miss_val,
            "duplicate_values": duplicated_value.item()
        }

        return result



        
            

        

