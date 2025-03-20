import dataclasses
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel, computed_field
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from ml_dcs.domain.mtsa import MTSAResult


class BaseMLInput(ABC, BaseModel):
    @classmethod
    @abstractmethod
    def init_by_mtsa_result(cls, mtsa_result: MTSAResult) -> "BaseMLInput":
        pass


# ===
# CALCULATION TIME
# ===
class MLCalculationTimePredictionInput1(BaseMLInput):
    total_number_of_states: int
    total_number_of_transitions: int
    total_number_of_controllable_actions: int
    total_number_of_uncontrollable_actions: int
    # ratio_of_controllable_actions: float
    number_of_models: int

    # milliseconds (ms)
    calculation_time: float

    @classmethod
    def init_by_mtsa_result(
        cls, mtsa_result: MTSAResult
    ) -> "MLCalculationTimePredictionInput1":
        variables = {
            "total_number_of_states": mtsa_result.compile_step.total_number_of_states,
            "total_number_of_transitions": mtsa_result.compile_step.total_number_of_transitions,
            "total_number_of_controllable_actions": mtsa_result.compile_step.total_number_of_controllable_actions,
            "total_number_of_uncontrollable_actions": mtsa_result.compile_step.total_number_of_uncontrollable_actions,
            "ratio_of_controllable_actions": mtsa_result.compile_step.ratio_of_controllable_actions,
            "number_of_models": mtsa_result.compile_step.number_of_models,
            "calculation_time": mtsa_result.duration_ms,
        }
        return cls(**variables)


class MLCalculationTimePredictionInput2(BaseMLInput):
    number_of_models_of_environments: int
    max_number_of_states_of_environments: int
    sum_of_number_of_transitions_of_environments: int
    sum_of_number_of_controllable_actions_of_environments: int
    sum_of_number_of_uncontrollable_actions_of_environments: int
    number_of_models_of_requirements: int
    sum_of_number_of_states_of_requirements: int
    sum_of_number_of_transitions_of_requirements: int
    sum_of_number_of_controllable_actions_of_requirements: int
    sum_of_number_of_uncontrollable_actions_of_requirements: int

    # milliseconds (ms)
    calculation_time: float

    @classmethod
    def init_by_mtsa_result(
        cls, mtsa_result: MTSAResult
    ) -> "MLCalculationTimePredictionInput2":
        variables = {
            "number_of_models_of_environments": mtsa_result.initial_models.number_of_models_of_environments,
            "max_number_of_states_of_environments": mtsa_result.initial_models.max_number_of_states_of_environments,
            "sum_of_number_of_transitions_of_environments": mtsa_result.initial_models.sum_of_number_of_transitions_of_environments,
            "sum_of_number_of_controllable_actions_of_environments": mtsa_result.initial_models.sum_of_number_of_controllable_actions_of_environments,
            "sum_of_number_of_uncontrollable_actions_of_environments": mtsa_result.initial_models.sum_of_number_of_uncontrollable_actions_of_environments,
            "number_of_models_of_requirements": mtsa_result.initial_models.number_of_models_of_requirements,
            "sum_of_number_of_states_of_requirements": mtsa_result.initial_models.sum_of_number_of_states_of_requirements,
            "sum_of_number_of_transitions_of_requirements": mtsa_result.initial_models.sum_of_number_of_transitions_of_requirements,
            "sum_of_number_of_controllable_actions_of_requirements": mtsa_result.initial_models.sum_of_number_of_controllable_actions_of_requirements,
            "sum_of_number_of_uncontrollable_actions_of_requirements": mtsa_result.initial_models.sum_of_number_of_uncontrollable_actions_of_requirements,
            "calculation_time": mtsa_result.duration_ms,
        }
        return cls(**variables)


# ===
# MEMORY USAGE
# ===
class MLMemoryUsagePredictionInput1(BaseMLInput):
    total_number_of_states: int
    total_number_of_transitions: int
    total_number_of_controllable_actions: int
    total_number_of_uncontrollable_actions: int
    # ratio_of_controllable_actions: float
    number_of_models: int

    # KiB
    memory_usage: int

    @classmethod
    def init_by_mtsa_result(
        cls, mtsa_result: MTSAResult
    ) -> "MLMemoryUsagePredictionInput1":
        variables = {
            "total_number_of_states": mtsa_result.compile_step.total_number_of_states,
            "total_number_of_transitions": mtsa_result.compile_step.total_number_of_transitions,
            "total_number_of_controllable_actions": mtsa_result.compile_step.total_number_of_controllable_actions,
            "total_number_of_uncontrollable_actions": mtsa_result.compile_step.total_number_of_uncontrollable_actions,
            "ratio_of_controllable_actions": mtsa_result.compile_step.ratio_of_controllable_actions,
            "number_of_models": mtsa_result.compile_step.number_of_models,
            "memory_usage": mtsa_result.max_memory_usage_kb,
        }
        return cls(**variables)


class MLMemoryUsagePredictionInput2(BaseMLInput):
    number_of_models_of_environments: int
    max_number_of_states_of_environments: int
    sum_of_number_of_transitions_of_environments: int
    sum_of_number_of_controllable_actions_of_environments: int
    sum_of_number_of_uncontrollable_actions_of_environments: int
    number_of_models_of_requirements: int
    sum_of_number_of_states_of_requirements: int
    sum_of_number_of_transitions_of_requirements: int
    sum_of_number_of_controllable_actions_of_requirements: int
    sum_of_number_of_uncontrollable_actions_of_requirements: int

    # KiB
    memory_usage: int

    @classmethod
    def init_by_mtsa_result(
        cls, mtsa_result: MTSAResult
    ) -> "MLMemoryUsagePredictionInput2":
        variables = {
            "number_of_models_of_environments": mtsa_result.initial_models.number_of_models_of_environments,
            "max_number_of_states_of_environments": mtsa_result.initial_models.max_number_of_states_of_environments,
            "sum_of_number_of_transitions_of_environments": mtsa_result.initial_models.sum_of_number_of_transitions_of_environments,
            "sum_of_number_of_controllable_actions_of_environments": mtsa_result.initial_models.sum_of_number_of_controllable_actions_of_environments,
            "sum_of_number_of_uncontrollable_actions_of_environments": mtsa_result.initial_models.sum_of_number_of_uncontrollable_actions_of_environments,
            "number_of_models_of_requirements": mtsa_result.initial_models.number_of_models_of_requirements,
            "sum_of_number_of_states_of_requirements": mtsa_result.initial_models.sum_of_number_of_states_of_requirements,
            "sum_of_number_of_transitions_of_requirements": mtsa_result.initial_models.sum_of_number_of_transitions_of_requirements,
            "sum_of_number_of_controllable_actions_of_requirements": mtsa_result.initial_models.sum_of_number_of_controllable_actions_of_requirements,
            "sum_of_number_of_uncontrollable_actions_of_requirements": mtsa_result.initial_models.sum_of_number_of_uncontrollable_actions_of_requirements,
            "memory_usage": mtsa_result.max_memory_usage_kb,
        }
        return cls(**variables)


# ===
# Results
# ===
class MLSimpleTrainingResult(BaseModel):
    algorithm: str
    random_state: int
    started_at: datetime
    finished_at: datetime

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._duration = None

    @computed_field
    @property
    def duration(self) -> timedelta:
        if self._duration is None:
            self._duration = self.finished_at - self.started_at
            return self._duration
        else:
            return self._duration


class MLSimpleTrainingResultSet(BaseModel):
    algorithm: str
    results: List[MLSimpleTrainingResult]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # === computed fields ===
        self._duration_sum = None
        self._duration_avg = None

    @computed_field
    @property
    def duration_sum(self) -> timedelta:
        if self._duration_sum is None:
            sum = 0
            for result in self.results:
                sum += result.duration.total_seconds()
                self._duration_sum = timedelta(seconds=sum)
            return self._duration_sum
        else:
            return self._duration_sum

    @computed_field
    @property
    def duration_avg(self) -> timedelta:
        if self._duration_avg is None:
            sum = self.duration_sum.total_seconds()
            avg = sum / len(self.results)
            self._duration_avg = timedelta(seconds=avg)
            return self._duration_avg
        else:
            return self._duration_avg


class MLSimpleTestingResult(BaseModel):
    algorithm: str
    random_state: int
    lts_names: List[str]
    actual_values: List[float]
    predicted_values: List[float]
    started_at: datetime
    finished_at: datetime

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._duration = None
        self._mae = None
        self._mse = None
        self._rmse = None
        self._mape = None
        self._r_squared = None
        self._actual_values_div1000 = None
        self._predicted_values_div1000 = None

    @computed_field
    @property
    def duration(self) -> timedelta:
        if self._duration is None:
            self._duration = self.finished_at - self.started_at
            return self._duration
        else:
            return self._duration

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
            self._mape = mean_absolute_percentage_error(
                self.actual_values, self.predicted_values
            )
        return self._mape

    @computed_field
    @property
    def r_squared(self) -> float:
        if self._r_squared is None:
            self._r_squared = r2_score(self.actual_values, self.predicted_values)
            return self._r_squared
        else:
            return self._r_squared

    @computed_field
    @property
    def actual_values_div1000(self) -> List[float]:
        if self._actual_values_div1000 is None:
            self._actual_values_div1000 = [value / 1000 for value in self.actual_values]
        return self._actual_values_div1000

    @computed_field
    @property
    def predicted_values_div1000(self) -> List[float]:
        if self._predicted_values_div1000 is None:
            self._predicted_values_div1000 = [
                value / 1000 for value in self.predicted_values
            ]
        return self._predicted_values_div1000


class MLSimpleTestingResultFinal(BaseModel):
    algorithm: str
    random_state: int
    duration: timedelta
    mae: float
    mse: float
    rmse: float
    r_squared: float


class MLSimpleTestingResultSet(BaseModel):
    algorithm: str
    results: List[MLSimpleTestingResultFinal]
    result_at_best_accuracy: MLSimpleTestingResult

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._duration_avg = None
        self._mae_variance = None
        self._r_squared_variance = None

    @computed_field
    @property
    def duration_avg(self) -> timedelta:
        if self._duration_avg is None:
            durations = []
            for result in self.results:
                durations.append(result.duration.total_seconds())
            s = sum(durations)
            avg = s / len(self.results)
            self._duration_avg = timedelta(seconds=avg)
            return self._duration_avg
        else:
            return self._duration_avg

    @computed_field
    @property
    def mae_variance(self) -> float:
        if self._mae_variance is None:
            mae_values = [result.mae for result in self.results]
            self._mae_variance = np.var(mae_values).item()
            return self._mae_variance
        else:
            return self._mae_variance

    @computed_field
    @property
    def r_squared_variance(self) -> float:
        if self._r_squared_variance is None:
            r_squared_values = [result.r_squared for result in self.results]
            self._r_squared_variance = np.var(r_squared_values).item()
            return self._r_squared_variance
        else:
            return self._r_squared_variance


@dataclasses.dataclass
class SimpleInputData:
    lts_names: List[str]
    x: np.ndarray
    y: pd.Series


class TrainingDataSet(SimpleInputData):
    pass


class TestingDataSet(SimpleInputData):
    pass
