from contextlib import asynccontextmanager

import pandas as pd

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.data.data_cleaning import clean_data
from src.inference.predict import Modelservice
from src.inference.validate import ValidateStudentData
from utils.load_config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading application services...")

    app.state.model_service = Modelservice()
    app.state.validator = ValidateStudentData()

    logger.info("Application services loaded.")

    yield

    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Student Success Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def health():
        return {
            "status": "healthy",
            "message": "Application is working fine.",
        }

    @app.post("/predict/file")
    async def predict_file(
        request: Request,
        file: UploadFile = File(...),
    ):
        try:
            logger.info("Reading uploaded CSV: %s", file.filename)

            df = pd.read_csv(file.file, sep=";")

            logger.info("Rows: %d Columns: %d", *df.shape)

            df = clean_data(df, config)

            validator = request.app.state.validator
            model_service = request.app.state.model_service

            df = validator.validate_csv_file(df)

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

        except Exception:
            logger.exception("Prediction failed.")
            raise HTTPException(
                status_code=500,
                detail="Internal server error.",
            )

    return app


app = create_app()