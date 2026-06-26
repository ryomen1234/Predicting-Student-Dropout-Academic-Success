from src.data.split import split_data
from utils.load_config import load_config

def test_data_split():

    config = load_config()

    X_train, X_test, y_train, y_test = split_data(config=config)
    print(f"X_train, y_train: {X_train.shape, y_train.shape} | X_test, y_test: {X_test.shape, y_test.shape}")

if __name__ == "__main__":
    test_data_split()