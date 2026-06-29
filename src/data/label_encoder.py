from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger

import pandas as pd

logger = get_logger(__name__)

def encode_label(
    y_train: pd.Series,
    y_test: pd.Series,
):
    logger.info("Encoding target labels")

    encoder = LabelEncoder()

    y_train_trf = encoder.fit_transform(y_train)
    y_test_trf = encoder.transform(y_test)

    logger.debug(
        "Class mapping: %s",
        dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    )

    logger.info(
        "Encoded %d classes",
        len(encoder.classes_)
    )

    return y_train_trf, y_test_trf, encoder

