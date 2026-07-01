from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import pandas as pd

from app.schema import StudentInput

app = FastAPI()


@app.get("/")
def health():
    return {"message": "application is working fine."}


@app.post("/predict_file")
async def upload_file(file: UploadFile = File(...)):
    
    df = pd.read_csv(file.file)

    return {
        "size": df.shape
    }