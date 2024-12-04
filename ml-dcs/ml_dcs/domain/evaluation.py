from datetime import datetime, timedelta
from typing import List

from pydantic import computed_field, BaseModel
from sklearn.metrics import mean_absolute_error, r2_score


class EvaluationResult(BaseModel):
    actual_values: List[float]
    predicted_values: List[float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mae = None
        self._r_squared = None

    @computed_field
    @property
    def mae(self) -> float:
        if self._mae is None:
            self._mae = mean_absolute_error(self.actual_values, self.predicted_values)
            return self._mae
        else:
            return self._mae

    @computed_field
    @property
    def r_squared(self) -> float:
        if self._r_squared is None:
            self._r_squared = r2_score(self.actual_values, self.predicted_values)
            return self._r_squared
        else:
            return self._r_squared
