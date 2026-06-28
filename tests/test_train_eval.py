from src.model import eval, train
from src.data import data_cleaning, split

from utils.load_config import load_config

def test_train_eval():
    config = load_config()

    X_train, X_test, y_train, y_test = split.split_data(config=config)

    X_train, X_test = data_cleaning.clean_data(X_test=X_test, X_train=X_train, config=config)

    model = train.train(X_train=X_train, y_train=y_train, config=config)
    score, report = eval.eval(model=model, X_test=X_test, y_test=y_test, config=config)

    print(f"score: {score}")
    print(report)

if __name__ == "__main__":
    test_train_eval()

   





    

