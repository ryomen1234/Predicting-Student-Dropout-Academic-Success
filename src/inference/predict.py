from src.inference.loader import ArtifactLoader
from utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

class Modelservice:
    def __init__(self) -> None:
        logger.info("Model loading....")
        self.model = ArtifactLoader.get_model()
        logger.info("Loading feature names....")
        self.le = ArtifactLoader.get_encoder()

    def predict(self, X: pd.DataFrame):
        pred =  self.model.predict(X)
        return self.le.inverse_transform(pred)
    

    

    