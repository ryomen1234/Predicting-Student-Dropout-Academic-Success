from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Any


class PreprocessingConfig(BaseModel):
    drop_columns: list[str]
    target_columns: list[str]
    target: str


class DataSplitConfig(BaseModel):
    test_size: float
    random_state: int


class DataPathConfig(BaseModel):
    data_dir_path: Path

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