import pandas as pd

from src.inference.predict import Modelservice
from src.inference.validate import ValidateStudentData
from src.data.data_cleaning import clean_data
from utils.io import load_csv
from utils.load_config import load_config


from pathlib import Path

def main():
    infer = Modelservice()
    config = load_config()

    test_data = load_csv(path=Path("data/raw/student_academic_data/data.csv"))
    # print(test_data.shape)
    test_data = clean_data(data=test_data, config=config)
    # print(test_data.shape)
    test_data = test_data.drop(columns=["Target"])
    # print(test_data.shape)

    vsd = ValidateStudentData()
    df = vsd.validate_csv_file(data=test_data)

    result = infer.predict(df)

    print(result)


if __name__ == "__main__":
    main()