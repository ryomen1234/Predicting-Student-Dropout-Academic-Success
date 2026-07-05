from src.data import data_cleaning
from src.data import split
from src.data.data_validation import DataValidation
from src.data.label_encoder import encode_label

from utils.io import (
    load_csv,
    save_csv,
    save_numpy,
    save_model,
    save_json,
)
from utils.logger import get_logger
from utils.load_config import load_config

logger = get_logger(__name__)


def run():
    logger.info("========== DATA PIPELINE STARTED ==========")

    config = load_config()
    logger.info("Config loaded")
    logger.debug(f"config: {config}")

    # Load raw data
    data_path = (
        config.data_paths.data_dir_path
        / "raw"
        / "student_academic_data"
        / "data.csv"
    )

    df = load_csv(data_path)
    logger.info("Data loaded.")

    # Validate
    validator = DataValidation(config)
    validation_result = validator.validate(df)

    # Clean
    df = data_cleaning.clean_data(
        data=df,
        config=config,
    )

    # Split
    X_train, X_test, y_train, y_test = split.split_data(
        config=config,
        df=df,
    )

    # Encode labels
    y_train, y_test, encoder = encode_label(
        y_train=y_train,
        y_test=y_test,
    )

    # Save processed data
    save_csv(
        X_train,
        config.processed_path.X_train_path,
    )

    save_csv(
        X_test,
        config.processed_path.X_test_path,
    )

    save_numpy(
        y_train,
        config.processed_path.y_train_path,
    )

    save_numpy(
        y_test,
        config.processed_path.y_test_path,
    )

    # Save encoder
    save_model(
        encoder,
        config.artifacts.encoder_path,
    )

    # Save validation report
    save_json(
        validation_result,
        config.artifacts.data_validation_path,
    )

    logger.info("========== DATA PIPELINE COMPLETED ==========")
