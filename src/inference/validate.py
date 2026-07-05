import pandas as pd

from src.inference.loader import ArtifactLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class ValidateStudentData:
    """Validate uploaded student data before inference."""

    def __init__(self) -> None:
        self.expected_features = ArtifactLoader.get_feature_names()

    def validate_csv_file(self, data: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Validating uploaded data.")

        # Validate input type
        if not isinstance(data, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        uploaded_columns = set(data.columns)
        expected_columns = set(self.expected_features)

        missing_columns = expected_columns - uploaded_columns
        extra_columns = uploaded_columns - expected_columns

        if missing_columns:
            logger.error("Missing columns: %s", sorted(missing_columns))
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if extra_columns:
            logger.error("Unexpected columns: %s", sorted(extra_columns))
            raise ValueError(
                f"Unexpected columns found: {sorted(extra_columns)}"
            )

        # Reorder columns to match training order
        validated_df = data[self.expected_features].copy()

        logger.info("Data validation completed successfully.")

        return validated_df