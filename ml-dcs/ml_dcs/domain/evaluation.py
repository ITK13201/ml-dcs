from enum import Enum
from typing import List

from pydantic import BaseModel, computed_field
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


class EvaluationTarget(Enum):
    CALCULATION_TIME = "Calculation Time"
    MEMORY_USAGE = "Memory Usage"


class EvaluationResult(BaseModel):
    actual_values: List[float]
    predicted_values: List[float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._actual_values_div1000 = None
        self._predicted_values_div1000 = None
        self._mae = None
        self._mse = None
        self._rmse = None
        self._r_squared = None

    @computed_field
    @property
    def actual_values_div1000(self) -> List[float]:
        if self._actual_values_div1000 is None:
            self._actual_values_div1000 = [
                actual_value / 1000 for actual_value in self.actual_values
            ]
        return self._actual_values_div1000

    @computed_field
    @property
    def predicted_values_div1000(self) -> List[float]:
        if self._predicted_values_div1000 is None:
            self._predicted_values_div1000 = [
                predicted_value / 1000 for predicted_value in self.predicted_values
            ]
        return self._predicted_values_div1000

    @computed_field
    @property
    def mae(self) -> float:
        if self._mae is None:
            self._mae = mean_absolute_error(self.actual_values, self.predicted_values)
        return self._mae

    @computed_field
    @property
    def mse(self) -> float:
        if self._mse is None:
            self._mse = mean_squared_error(self.actual_values, self.predicted_values)
        return self._mse

    @computed_field
    @property
    def rmse(self) -> float:
        if self._rmse is None:
            self._rmse = root_mean_squared_error(
                self.actual_values, self.predicted_values
            )
        return self._rmse

    @computed_field
    @property
    def r_squared(self) -> float:
        if self._r_squared is None:
            self._r_squared = r2_score(self.actual_values, self.predicted_values)
        return self._r_squared
