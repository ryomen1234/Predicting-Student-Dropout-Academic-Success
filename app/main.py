from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import pandas as pd

from src.inference.predict import Modelservice
from src.inference.validate import ValidateStudentData
from src.data.data_cleaning import clean_data
from utils.load_config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Student Success Prediction API",
    version="1.0.0",
)

# Load artifacts once
config = load_config()
model_service = Modelservice()
validator = ValidateStudentData()


@app.get("/")
def health():
    return {
        "status": "healthy",
        "message": "Application is working fine."
    }


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """
    Predict student outcomes from a CSV file.
    """
    try:
        logger.info("Reading uploaded CSV: %s", file.filename)

        df = pd.read_csv(file.file, sep=";")

        logger.info("Rows: %d Columns: %d", *df.shape)

        # Apply same preprocessing used during training
        df = clean_data(df, config)

        # Target column should not be present during inference
        if "Target" in df.columns:
            df = df.drop(columns=["Target"])

        # Validate features
        df = validator.validate_csv_file(df)

        # Predict
        predictions = model_service.predict(df)

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "rows": len(predictions),
                "predictions": predictions,
            },
        )

    except ValueError as e:
        logger.exception("Validation failed.")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )