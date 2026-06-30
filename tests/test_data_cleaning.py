from src.data.data_cleaning import clean_data
from src.data.split import split_data
from utils.load_config import load_config
from pathlib import Path
from utils.io import save_model, load_csv

def test_data_cleaning():
    config = load_config()
    data_path = config.data_paths.data_dir_path / "raw" / "student_academic_data" / "data.csv"
    df = load_csv(path=data_path)

    X_train, X_test, _, _ = split_data(config=config, df=df)
    print(f"before data cleaning X_train adn X_test shape: {X_train.shape}, {X_test.shape}")

    X_train, X_test = clean_data(X_test=X_test, X_train=X_train, config=config)
    print(f"after data cleaning X_train adn X_test shape: {X_train.shape}, {X_test.shape}")

    features_name_path = Path("artifacts/feature_names.joblib") 
    save_model(X_train.columns.tolist(), path=features_name_path)


if __name__ == "__main__":
    test_data_cleaning()
