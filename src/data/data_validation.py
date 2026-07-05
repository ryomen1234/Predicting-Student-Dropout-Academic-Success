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
        self.target_col = config.preprocessing.target_col
        self.target = config.preprocessing.target

    def validate(self, df):
        logger.info("Data validation started.")


        data_shape = df.shape
        logger.debug(f"data shape: {data_shape}")

        missing_values = {
        column: int(count)
        for column, count in df.isnull().sum().items()
        if count > 0}
        logger.debug(f"missing values: {missing_values}")

        duplicated_value = int(df.duplicated().sum())
        logger.debug(f"duplicated values : {duplicated_value}")


        if set(df[self.target_col].unique()) != set(self.target):
            logger.error(f"target column {self.target} mismatch")
            raise ValueError("target column don't match: {self.target}")
        
        result = {
            "data_shape": data_shape,
            "missing_values": missing_values,
            "duplicate_values": duplicated_value
        }

        return result



        
            

        

