from pydantic import BaseModel


class MLCalculationTimePredictionInput(BaseModel):
    calculation_time: float
