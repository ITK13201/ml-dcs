from enum import Enum
from typing import List

import pandas as pd
from pydantic import BaseModel, computed_field
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from ml_dcs.domain.ml_gnn import GNNTestingResult
from ml_dcs.domain.ml_simple import MLSimpleTestingResultSet


class EvaluationTarget(Enum):
    CALCULATION_TIME = "Calculation Time"
    MEMORY_USAGE = "Memory Usage"


class EvaluationResult(BaseModel):
    actual_values: List[float]
    predicted_values: List[float]
    lts_names: List[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._actual_values_div1000 = None
        self._predicted_values_div1000 = None
        self._mae = None
        self._mse = None
        self._rmse = None
        self._mape = None
        self._r_squared = None

    @classmethod
    def from_simple_class(cls, result: MLSimpleTestingResultSet):
        return cls(
            actual_values=result.result_at_best_accuracy.actual_values,
            predicted_values=result.result_at_best_accuracy.predicted_values,
            lts_names=result.result_at_best_accuracy.lts_names,
        )

    @classmethod
    def from_gnn_class(cls, result: GNNTestingResult):
        return cls(
            actual_values=result.actual_values,
            predicted_values=result.predicted_values,
            lts_names=result.lts_names,
        )

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
    def mape(self) -> float:
        if self._mape is None:
            self._mape = (
                mean_absolute_percentage_error(
                    self.actual_values, self.predicted_values
                )
                * 100
            )
        return self._mape

    @computed_field
    @property
    def r_squared(self) -> float:
        if self._r_squared is None:
            self._r_squared = r2_score(self.actual_values, self.predicted_values)
        return self._r_squared

    @property
    def dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "lts": self.lts_names,
                "actual_values": self.actual_values,
                "predicted_values": self.predicted_values,
            }
        )
        df = df.sort_values(by="actual_values", ascending=True)
        return df

    @property
    def dataframe_div1000(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "lts": self.lts_names,
                "actual_values": self.actual_values_div1000,
                "predicted_values": self.predicted_values_div1000,
            }
        )
        df = df.sort_values(by="actual_values", ascending=True)
        return df
