from src.data.data_cleaning import clean_data
from src.data.split import split_data
from utils.load_config import load_config

def test_data_cleaning():
    config = load_config()

    X_train, X_test, _, _ = split_data(config=config)
    print(f"before data cleaning X_train adn X_test shape: {X_train.shape}, {X_test.shape}")

    X_train, X_test = clean_data(X_test=X_test, X_train=X_train, config=config)
    print(f"after data cleaning X_train adn X_test shape: {X_train.shape}, {X_test.shape}")


if __name__ == "__main__":
    test_data_cleaning()
