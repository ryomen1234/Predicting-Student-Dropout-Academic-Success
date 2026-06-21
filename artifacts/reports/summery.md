# Exploratory Data Analysis Summary

## Dataset Overview

* Total records: **4,424**
* Total features: **36**
* Prediction task: Multi-class classification

  * Dropout
  * Enrolled
  * Graduate

## Data Quality

* No missing values were found.
* No duplicate records were detected.
* Column names were cleaned to remove hidden whitespace and tab characters.

## Target Distribution

The dataset is moderately imbalanced:

* Graduate: **49.93%**
* Dropout: **32.12%**
* Enrolled: **17.95%**

This imbalance will be considered during model training.

## Key Findings

* Academic performance in the first and second semesters is the strongest indicator of final student outcome.
* Financial factors such as **Tuition fees up to date** and **Debtor** have a strong relationship with dropout.
* **Scholarship holder** also shows a positive association with graduation.
* Demographic and economic variables generally have lower predictive power than academic and financial features.

## Feature Selection

Feature selection was performed using:

* Statistical tests (Chi-Square and Kruskal-Wallis)
* Random Forest feature importance
* Mutual Information
* Variance Inflation Factor (VIF)

Based on these analyses:

* Highly redundant first-semester academic features were removed.
* Low-information features such as **Nationality**, **Educational special needs**, **International**, and **Inflation rate** were removed.
* The final feature set contains **26 features** for model training.

## Preprocessing Decisions

* Remove redundant and low-information features.
* Encode categorical variables.
* Scale numerical features where required by the selected model.
* Address class imbalance during model training if necessary.

## Conclusion

The dataset is clean and suitable for machine learning without major preprocessing challenges. Academic performance and financial status are the strongest predictors of student success, while several demographic variables contribute relatively little information. The selected feature set provides a compact and informative representation for building robust predictive models.
