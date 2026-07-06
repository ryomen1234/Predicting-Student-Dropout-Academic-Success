from src.data import data_cleaning
from src.data import split
from src.data.data_validation import DataValidation
from src.data.label_encoder import encode_label

from utils.io import (
    save_json,
    save_csv,
    save_numpy,
    load_csv,
    save_model
)

from utils.logger import get_logger
from utils.load_config import load_config


logger = get_logger(__name__)

def run():
    logger.info("======== Running  DataPipeline ==========")
   
    # load config
    config = load_config()
    logger.info("config file loaded")

    # laod data
    data_path = config.data_paths.data_dir_path / "raw" / "student_academic_data" / "data.csv"
    df = load_csv(path=data_path)

    # validate dataset
    dv = DataValidation(config=config)
    validation_result = dv.validate(df)

    # clean data
    df = data_cleaning.clean_data(data=df, config=config)

    # split into train and test
    X_train, X_test, y_train, y_test = split.split_data(config=config, df=df)

    # label encoder
    y_train, y_test, encoder = encode_label(y_train=y_train, y_test=y_test)

    save_json(data=validation_result, path=config.artifacts.data_validation_path)

    save_model(model=encoder, path=config.artifacts.encoder_path)

    save_csv(data=X_train, path=config.processed_data.X_train_path)
    save_csv(data=X_test, path=config.processed_data.X_test_path)

    save_numpy(data=y_train, path=config.processed_data.y_train_path)
    save_numpy(data=y_test, path=config.processed_data.y_test_path)

    logger.info("======= DataPipeline run successfully =========")

if __name__ == "__main__":
    run()










