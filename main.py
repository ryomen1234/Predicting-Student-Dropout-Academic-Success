from src.data import data_cleaning
from src.data import split
from src.data.data_validation import DataValidation
from src.data.label_encoder import encode_label

from src.model import eval, train 

from utils.io import save_csv, save_model, save_json, load_csv
from utils.logger import get_logger
from utils.load_config import load_config

from pathlib import Path

logger = get_logger(__name__)

def run():
    logger.info("running main pipeline")
   
    # load config
    config = load_config()
    logger.info("config file loaded")
    logger.debug(config)

    # laod data
    data_path = config.data_paths.data_dir_path / "raw" / "student_academic_data" / "data.csv"
    df = load_csv(path=data_path)

    # validate dataset
    dv = DataValidation(config=config)
    validation_result = dv.validate(df)

    # split into train and test
    X_train, X_test, y_train, y_test = split.split_data(config=config, df=df)

    # label encoder
    y_train, y_test, encoder = encode_label(y_train=y_train, y_test=y_test)

    # clean data
    X_train, X_test = data_cleaning.clean_data(X_train=X_train, X_test=X_test, config=config)

    # model training
    model = train.train_model(X_train=X_train,
                              y_train=y_train,
                              config=config)
    
    # model evaluation 
    result, report = eval.eval(
        model=model,
        X_test=X_test,
        y_test=y_test,
        config=config
    )

    # save artifacts and results
    # paths
    save_dir = Path("artifacts")

    model_path = save_dir / "models" / "best_model.joblib"
    encoder_path = save_dir / "encoder.joblib"
    data_validation_result = save_dir / "data_validation_result.json"
    eval_report = save_dir / "eval_result.joblib"
    eval_result = save_dir / "eval_result.json" 

    save_json(data=result, path=eval_result)
    save_json(data=validation_result, path=data_validation_result)

    save_model(model=model, path=model_path)
    save_model(model=encoder, path=encoder_path)
    save_model(model=report, path=eval_report)


    logger.info("Pipeline successfully run")

if __name__ == "__main__":
    run()










