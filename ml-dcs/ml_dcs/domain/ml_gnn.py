import dataclasses
from datetime import datetime, timedelta
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, computed_field
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from torch import Tensor
from torch_geometric.data import Batch


@dataclasses.dataclass
class GNNInputData:
    lts_name: str
    lts_set_graph: Batch
    target: Tensor


class TrainingData(GNNInputData):
    pass


class ValidationData(GNNInputData):
    pass


class TestingData(GNNInputData):
    pass


class GNNTrainingResultEpoch(BaseModel):
    training_loss_avg: float
    validation_loss_avg: float
    started_at: datetime
    finished_at: datetime

    def __init__(self, *args, **kwargs) -> None:
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


class GNNTrainingResult(BaseModel):
    epoch_results: List[GNNTrainingResultEpoch]
    layer_num: int
    started_at: datetime
    finished_at: datetime

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epochs = None
        self._duration = None
        self._epoch_duration_avg = None

    @computed_field
    @property
    def epochs(self) -> int:
        if self._epochs is None:
            self._epochs = len(self.epoch_results)
            return self._epochs
        else:
            return self._epochs

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
    def epoch_duration_avg(self) -> timedelta:
        if self._epoch_duration_avg is None:
            durations = []
            for epoch in self.epoch_results:
                durations.append(epoch.duration.total_seconds())
            s = sum(durations)
            avg = s / self.epochs
            self._epoch_duration_avg = timedelta(seconds=avg)
            return self._epoch_duration_avg
        else:
            return self._epoch_duration_avg


class GNNTestingResultTask(BaseModel):
    lts_name: str
    loss: float
    actual: float
    predicted: float
    started_at: datetime
    finished_at: datetime

    def __init__(self, *args, **kwargs) -> None:
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


class GNNTestingResult(BaseModel):
    task_results: List[GNNTestingResultTask]
    layer_num: int

    model_config = ConfigDict(frozen=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # === properties ===
        self._task_results_length = None
        self._loss_sum = None
        self._lts_names = None
        self._actual_values = None
        self._predicted_values = None
        self._actual_values_div1000 = None
        self._predicted_values_div1000 = None

        # === computed fields ===
        self._loss_avg = None
        self._mae = None
        self._mse = None
        self._rmse = None
        self._mape = None
        self._r_squared = None
        self._duration_avg = None

    @property
    def task_results_length(self) -> int:
        if self._task_results_length is None:
            self._task_results_length = len(self.task_results)
            return self._task_results_length
        else:
            return self._task_results_length

    @property
    def loss_sum(self) -> float:
        if self._loss_sum is None:
            self._loss_sum = 0
            for result in self.task_results:
                self._loss_sum += result.loss
            return self._loss_sum
        else:
            return self._loss_sum

    @property
    def lts_names(self) -> List[str]:
        if self._lts_names is None:
            self._lts_names = [task.lts_name for task in self.task_results]
            return self._lts_names
        else:
            return self._lts_names

    @property
    def actual_values(self) -> List[float]:
        if self._actual_values is None:
            self._actual_values = [task.actual for task in self.task_results]
            return self._actual_values
        else:
            return self._actual_values

    @property
    def predicted_values(self) -> List[float]:
        if self._predicted_values is None:
            self._predicted_values = [task.predicted for task in self.task_results]
            return self._predicted_values
        else:
            return self._predicted_values

    @property
    def actual_values_div1000(self) -> List[float]:
        if self._actual_values_div1000 is None:
            self._actual_values_div1000 = [value / 1000 for value in self.actual_values]
        return self._actual_values_div1000

    @property
    def predicted_values_div1000(self) -> List[float]:
        if self._predicted_values_div1000 is None:
            self._predicted_values_div1000 = [
                value / 1000 for value in self.predicted_values
            ]
        return self._predicted_values_div1000

    @computed_field
    @property
    def loss_avg(self) -> float:
        if self._loss_avg is None:
            self._loss_avg = self.loss_sum / self.task_results_length
            return self._loss_avg
        else:
            return self._loss_avg

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
        else:
            return self._r_squared

    @computed_field
    @property
    def duration_avg(self) -> timedelta:
        if self._duration_avg is None:
            durations = []
            for result in self.task_results:
                durations.append(result.duration.total_seconds())
            s = sum(durations)
            avg = s / self.task_results_length
            self._duration_avg = timedelta(seconds=avg)
            return self._duration_avg
        else:
            return self._duration_avg


# ===
# Preprocessing
# ===
class NodeFeature(BaseModel):
    input_model_type: Literal[-1, 1]
    input_model_number: float
    state_number: float
    is_not_error_state: Literal[-1, 1]
    is_start_state: Literal[0, 1]

    def get_array(self):
        return [
            self.input_model_type,
            self.input_model_number,
            self.state_number,
            self.is_not_error_state,
            self.is_start_state,
        ]


class EdgeAttribute(BaseModel):
    action: float
    is_controllable: Literal[0, 1]
    is_forward: Literal[-1, 1]

    def get_array(self):
        return [
            self.action,
            self.is_controllable,
            self.is_forward,
        ]


class ModelFeature(BaseModel):
    node_features: List[List[float]]
    edge_indexes: List[List[int]]
    edge_attributes: List[List[float]]
