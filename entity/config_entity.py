from pathlib import Path
from pydantic import BaseModel


class PreprocessingConfig(BaseModel):
    drop_columns: list[str]


class DataSplitConfig(BaseModel):
    test_size: float
    random_state: int


class DataPathConfig(BaseModel):
    data_dir_path: Path


class Config(BaseModel):
    preprocessing: PreprocessingConfig
    data_split: DataSplitConfig
    data_paths: DataPathConfig