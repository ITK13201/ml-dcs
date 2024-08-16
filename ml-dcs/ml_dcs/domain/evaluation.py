from datetime import timedelta
from typing import List

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


class EvaluationResultBestAccuracyScores(BaseModel):
    random_state: int
    r2_score: float
    mae_score: float

    model_config = ConfigDict(frozen=True)


class EvaluationResult(BaseModel):
    ml_algorithm: str
    ml_input_class: str
    random_states: List[int]
    r2_scores: List[float]
    mae_scores: List[float]
    r2_score_variance: float
    prediction_time: timedelta
    best_accuracy_scores: EvaluationResultBestAccuracyScores

    @field_validator("prediction_time", mode="before", check_fields=True)
    @classmethod
    def validate_calculation_duration(cls, value: float) -> timedelta:
        return timedelta(seconds=value)

    @field_serializer("prediction_time")
    def serialize_prediction_time(self, value: timedelta) -> str:
        return str(value)

    model_config = ConfigDict(frozen=True)
