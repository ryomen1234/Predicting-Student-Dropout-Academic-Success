import pandas as pd

from src.inference.predict import Inference


def main():

    infer = Inference()

    sample = pd.DataFrame(
        [
            {
                "Marital status": 1,
                "Application mode": 17,
                "Application order": 1,
                "Course": 171,
                "Previous qualification": 1,
                "Previous qualification (grade)": 140.0,
                "Mother's qualification": 19,
                "Father's qualification": 19,
                "Mother's occupation": 5,
                "Father's occupation": 5,
                "Admission grade": 142.5,
                "Debtor": 0,
                "Tuition fees up to date": 1,
                "Gender": 1,
                "Scholarship holder": 1,
                "Age at enrollment": 18,
                "Curricular units 1st sem (enrolled)": 6,
                "Curricular units 1st sem (evaluations)": 6,
                "Curricular units 1st sem (approved)": 6,
                "Curricular units 1st sem (grade)": 14.5,
                "Inflation rate": 1.4,
            }
        ]
    )

    result = infer.predict(sample)

    print(result)


if __name__ == "__main__":
    main()