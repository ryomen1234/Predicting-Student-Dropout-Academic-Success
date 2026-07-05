from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Any


class PreprocessingConfig(BaseModel):
    drop_columns: list[str]
    target_col: str
    target: list[str]


class DataSplitConfig(BaseModel):
    test_size: float
    random_state: int


class DataPathConfig(BaseModel):
    data_dir_path: Path

class ArtifactsPathConfig(BaseModel):
    model_path: Path 
    encoder_path: Path 
    feature_names_path: Path
    data_validation_path: Path
    evaluation_path: Path
    classification_report_path: Path

class ProcessedDataPathConfig(BaseModel):
    X_train_path: Path
    X_test_path: Path
    y_train_path: Path
    y_test_path: Path

class ModelConfig(BaseModel):
    model_name: str 
    best_params: Dict[str, Any]
    random_state: int 
    n_splits: int

class Config(BaseModel):
    preprocessing: PreprocessingConfig
    data_split: DataSplitConfig
    data_paths: DataPathConfig
    model: ModelConfig
    artifacts: ArtifactsPathConfig
    processed_path: ProcessedDataPathConfig