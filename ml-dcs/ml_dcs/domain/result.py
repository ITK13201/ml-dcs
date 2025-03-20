import json
from typing import List

from pydantic import BaseModel, computed_field

from ml_dcs.domain.ml_gnn import GNNTestingResult, GNNTrainingResult
from ml_dcs.domain.ml_simple import MLSimpleTestingResultSet, MLSimpleTrainingResultSet


class UnknownScenarioResult(BaseModel):
    list_epochs: List[int] = []
    list_mae: List[float] = []
    list_rmse: List[float] = []
    list_mape: List[float] = []
    list_t_train: List[float] = []
    list_t_pred: List[float] = []

    @computed_field
    @property
    def epochs(self) -> float | None:
        if self.list_epochs:
            return sum(self.list_epochs) / len(self.list_epochs)
        else:
            return None

    @computed_field
    @property
    def mae(self) -> float | None:
        if self.list_mae:
            return sum(self.list_mae) / len(self.list_mae)
        else:
            return None

    @computed_field
    @property
    def rmse(self) -> float | None:
        if self.list_rmse:
            return sum(self.list_rmse) / len(self.list_rmse)
        else:
            return None

    @computed_field
    @property
    def mape(self) -> float | None:
        if self.list_mape:
            return sum(self.list_mape) / len(self.list_mape)
        else:
            return None

    @computed_field
    @property
    def t_train(self) -> float | None:
        if self.list_t_train:
            return sum(self.list_t_train) / len(self.list_t_train)
        else:
            return None

    @computed_field
    @property
    def t_pred(self) -> float | None:
        if self.list_t_pred:
            return sum(self.list_t_pred) / len(self.list_t_pred)
        else:
            return None

    def update_with_simple_from_testing(self, input_path: str):
        with open(input_path, "r") as f:
            data_dict = json.load(f)
        data_model = MLSimpleTestingResultSet(**data_dict)
        self.list_mae.append(data_model.result_at_best_accuracy.mae / 1000)
        self.list_rmse.append(data_model.result_at_best_accuracy.rmse / 1000)
        self.list_mape.append(data_model.result_at_best_accuracy.mape)
        self.list_t_pred.append(
            data_model.result_at_best_accuracy.duration.total_seconds() * 1000
        )

    def update_with_gnn_from_testing(self, input_path: str):
        with open(input_path, "r") as f:
            data_dict = json.load(f)
        data_model = GNNTestingResult(**data_dict)
        self.list_mae.append(data_model.mae / 1000)
        self.list_rmse.append(data_model.rmse / 1000)
        self.list_mape.append(data_model.mape)
        self.list_t_pred.append(data_model.duration_avg.total_seconds() * 1000)

    def update_with_simple_from_training(self, input_path: str):
        with open(input_path, "r") as f:
            data_dict = json.load(f)
        data_model = MLSimpleTrainingResultSet(**data_dict)
        self.list_t_train.append(data_model.duration_avg.total_seconds())

    def update_with_gnn_from_training(self, input_path: str):
        with open(input_path, "r") as f:
            data_dict = json.load(f)
        data_model = GNNTrainingResult(**data_dict)
        self.list_epochs.append(data_model.epochs)
        self.list_t_train.append(data_model.duration.total_seconds())
