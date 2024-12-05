from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ml_dcs.domain.evaluation import EvaluationTarget
from ml_dcs.domain.ml_simple import TestingDataSet, TrainingDataSet
from ml_dcs.domain.mtsa import MTSAResult
from ml_dcs.internal.preprocessor.preprocessor import MLSimplePreprocessor

DEFAULT_RANDOM_STATE = 42


class MLSimpleDataUtil:
    def __init__(
        self,
        mtsa_results: List[MTSAResult],
        target: EvaluationTarget,
        threshold: float = None,
    ):
        # args
        self.mtsa_results = mtsa_results
        self.target = target
        self.threshold = threshold
        # additional
        if self.threshold is not None:
            self.mtsa_results = self._exclude_by_threshold()
        (
            self.training_mtsa_results,
            self.testing_mtsa_results,
        ) = self._split_mtsa_results(self.mtsa_results)
        self.preprocessor = MLSimplePreprocessor(
            self.training_mtsa_results,
            self.testing_mtsa_results,
            self.target,
        )
        self.training_dataset, self.testing_dataset = self.preprocessor.preprocess()

    def _exclude_by_threshold(self) -> List[MTSAResult]:
        updated = []
        for mtsa_result in self.mtsa_results:
            match self.target:
                case EvaluationTarget.CALCULATION_TIME:
                    if mtsa_result.duration_ms > self.threshold:
                        continue
                case EvaluationTarget.MEMORY_USAGE:
                    if mtsa_result.max_memory_usage_kb > self.threshold:
                        continue
                case _:
                    raise ValueError(f"Not supported: {self.target}")
            updated.append(mtsa_result)
        return updated

    def _split_mtsa_results(
        self, mtsa_results: List[MTSAResult]
    ) -> Tuple[List[MTSAResult], List[MTSAResult]]:
        training_results, tmp = train_test_split(
            mtsa_results, test_size=0.3, random_state=DEFAULT_RANDOM_STATE
        )
        _, testing_results = train_test_split(
            tmp, test_size=0.5, random_state=DEFAULT_RANDOM_STATE
        )
        return training_results, testing_results

    def get_training_dataset(self) -> TrainingDataSet:
        return self.training_dataset

    def get_testing_dataset(self) -> TestingDataSet:
        return self.testing_dataset


class RegressionAlgorithm(Enum):
    LINEAR_REGRESSION = "Linear Regression"
    RANDOM_FOREST = "Random Forest"
    GRADIENT_BOOSTING_DECISION_TREE = "Gradient Boosting Decision Tree"
    DECISION_TREE = "Decision Tree"
    LOGISTIC_REGRESSION = "Logistic Regression"


class RegressionModel:
    def __init__(
        self, algorithm: RegressionAlgorithm, random_state=DEFAULT_RANDOM_STATE
    ):
        self.model = None
        self.algorithm = algorithm
        self.random_state = random_state

    def train(self, x_train_std: np.ndarray, y_train: pd.Series):
        match self.algorithm:
            case RegressionAlgorithm.LINEAR_REGRESSION:
                self.model = LinearRegression()
            case RegressionAlgorithm.RANDOM_FOREST:
                self.model = RandomForestRegressor(random_state=self.random_state)
            case RegressionAlgorithm.GRADIENT_BOOSTING_DECISION_TREE:
                self.model = GradientBoostingRegressor(random_state=self.random_state)
            case RegressionAlgorithm.DECISION_TREE:
                self.model = DecisionTreeRegressor(random_state=self.random_state)
            case RegressionAlgorithm.LOGISTIC_REGRESSION:
                self.model = LogisticRegression(
                    random_state=self.random_state, max_iter=10**4
                )
            case _:
                raise ValueError(f"Unsupported algorithm {self.algorithm}")
        self.model.fit(x_train_std, y_train)

    def predict(self, x_test_std: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError(f"Model is not fitted yet.")
        return self.model.predict(x_test_std)
