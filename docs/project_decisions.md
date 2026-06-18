Problem statement:
"The goal was to build a multi-class classification model that predicts whether a student will graduate, remain enrolled, or drop out. The model uses demographic, socio-economic, enrollment, and first-year academic performance data. Since the dataset is imbalanced, I evaluated models using macro F1-score and class-wise metrics rather than relying solely on accuracy. The business objective is to identify at-risk students early and enable targeted interventions to reduce dropout rates."

Remove (Leakage Features)

These directly measure academic performance after enrollment:

Curricular units 1st sem (credited)
Curricular units 1st sem (enrolled)
Curricular units 1st sem (evaluations)
Curricular units 1st sem (approved)
Curricular units 1st sem (grade)
Curricular units 1st sem (without evaluations)
Curricular units 2nd sem (credited)
Curricular units 2nd sem (enrolled)
Curricular units 2nd sem (evaluations)
Curricular units 2nd sem (approved)
Curricular units 2nd sem (grade)
Curricular units 2nd sem (without evaluations)

Reason: These variables are almost direct indicators of success/failure. Using them to predict the final outcome is like using the answer to predict the answer.


any null values: false
any duplicated values: false

TARGET DISTRIBUSTION: 
Target
Graduate    2209
Dropout     1421
Enrolled     794

-dataset is imbalance
-Graduate class is dominating







