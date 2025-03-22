import json
from typing import List

import numpy as np
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
    weight_by_scenario: List[int] = []

    @computed_field
    @property
    def epochs(self) -> float | None:
        if self.list_epochs:
            return np.average(self.list_epochs, weights=self.weight_by_scenario).astype(
                float
            )
        else:
            return None

    @computed_field
    @property
    def mae(self) -> float | None:
        if self.list_mae:
            return np.average(self.list_mae, weights=self.weight_by_scenario).astype(
                float
            )
        else:
            return None

    @computed_field
    @property
    def rmse(self) -> float | None:
        if self.list_rmse:
            return np.average(self.list_rmse, weights=self.weight_by_scenario).astype(
                float
            )
        else:
            return None

    @computed_field
    @property
    def mape(self) -> float | None:
        if self.list_mape:
            return np.average(self.list_mape, weights=self.weight_by_scenario).astype(
                float
            )
        else:
            return None

    @computed_field
    @property
    def t_train(self) -> float | None:
        if self.list_t_train:
            return np.average(
                self.list_t_train, weights=self.weight_by_scenario
            ).astype(float)
        else:
            return None

    @computed_field
    @property
    def t_pred(self) -> float | None:
        if self.list_t_pred:
            return np.average(self.list_t_pred, weights=self.weight_by_scenario).astype(
                float
            )
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
        self.weight_by_scenario.append(data_model.result_at_best_accuracy.size)

    def update_with_gnn_from_testing(self, input_path: str):
        with open(input_path, "r") as f:
            data_dict = json.load(f)
        data_model = GNNTestingResult(**data_dict)
        self.list_mae.append(data_model.mae / 1000)
        self.list_rmse.append(data_model.rmse / 1000)
        self.list_mape.append(data_model.mape)
        self.list_t_pred.append(data_model.duration_avg.total_seconds() * 1000)
        self.weight_by_scenario.append(data_model.task_results_length)

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
