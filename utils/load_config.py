from pathlib import Path 
import yaml 

from entity.config_entity import Config
from utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def get_config():
  
    try:
        with open(CONFIG_DIR / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.debug("config file loaded.")
    except FileNotFoundError as e:
        logger.error(f"File {CONFIG_DIR / "config.yaml"} not found")
        raise FileNotFoundError
    
    return Config(**config)




