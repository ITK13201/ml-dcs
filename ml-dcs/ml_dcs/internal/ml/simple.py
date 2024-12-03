import datetime
from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

DEFAULT_RANDOM_STATE = 42


class MLSimpleDataUtil:
    def _preprocess(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        # split dataset to train and test
        x_train, x_tmp, y_train, y_tmp = train_test_split(
            df.iloc[:, :-1],
            df.iloc[:, -1],
            test_size=0.3,
            random_state=DEFAULT_RANDOM_STATE,
        )
        _, x_test, _, y_test = train_test_split(
            x_tmp,
            y_tmp,
            test_size=0.5,
            random_state=DEFAULT_RANDOM_STATE,
        )
        # standardize
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_std: np.ndarray = scaler.transform(x_train)
        x_test_std: np.ndarray = scaler.transform(x_test)
        return x_train_std, x_test_std, y_train, y_test

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.x_train, self.x_test, self.y_train, self.y_test = self._preprocess(df)

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        return self.x_train, self.x_test, self.y_train, self.y_test


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
