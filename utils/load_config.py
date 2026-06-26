from pathlib import Path 
import yaml 

from entity.config_entity import Config
from utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def load_config():

    config_file_path = CONFIG_DIR / "config.yaml"
  
    try:
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        logger.debug("config file loaded.")
    except FileNotFoundError as e:
        logger.error(f"File {config_file_path} not found")
        raise FileNotFoundError(f"{config_file_path} not found.")
    
    return Config(**config)




