from src.data.data_validation import DataValidation
from utils.load_config import load_config

def test_data_validation():

    config = load_config()

    d = DataValidation(config=config)

    result = d.main()

    print(result)

if __name__ == "__main__":
    test_data_validation()