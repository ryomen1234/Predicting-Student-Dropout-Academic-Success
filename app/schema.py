from pydantic import BaseModel, Field, ConfigDict


class StudentInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    marital_status: int = Field(alias="Marital status")
    application_mode: int = Field(alias="Application mode")
    application_order: int = Field(alias="Application order")
    course: int = Field(alias="Course")
    previous_qualification: int = Field(alias="Previous qualification")
    previous_qualification_grade: float = Field(
        alias="Previous qualification (grade)"
    )

    mothers_qualification: int = Field(alias="Mother's qualification")
    fathers_qualification: int = Field(alias="Father's qualification")

    mothers_occupation: int = Field(alias="Mother's occupation")
    fathers_occupation: int = Field(alias="Father's occupation")

    admission_grade: float = Field(alias="Admission grade")

    debtor: int = Field(alias="Debtor")
    tuition_fees_up_to_date: int = Field(alias="Tuition fees up to date")
    gender: int = Field(alias="Gender")
    scholarship_holder: int = Field(alias="Scholarship holder")

    age_at_enrollment: int = Field(alias="Age at enrollment")

    curricular_units_1st_sem_enrolled: int = Field(
        alias="Curricular units 1st sem (enrolled)"
    )

    curricular_units_1st_sem_evaluations: int = Field(
        alias="Curricular units 1st sem (evaluations)"
    )

    curricular_units_1st_sem_approved: int = Field(
        alias="Curricular units 1st sem (approved)"
    )

    curricular_units_1st_sem_grade: float = Field(
        alias="Curricular units 1st sem (grade)"
    )

    inflation_rate: float = Field(alias="Inflation rate")