from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml_dcs.domain.ml import MLResult


class BasePrediction:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.x_train: Optional[pd.DataFrame] = None
        self.x_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.x_train_std: Optional[pd.DataFrame] = None
        self.x_test_std: Optional[np.ndarray] = None
        self.predicted: Optional[np.ndarray] = None

    def _preprocess(self):
        # create train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df.iloc[:, :-1], self.df.iloc[:, -1], test_size=0.2, random_state=42
        )

        # standardize
        scaler = StandardScaler()
        scaler.fit(self.x_train)
        self.x_train_std = scaler.transform(self.x_train)
        self.x_test_std: pd.DataFrame = scaler.transform(self.x_test)

    def _grant_upper_limit(self, limit: int):
        removed_x_indexes = []
        removed_y_indexes = []
        count = 0
        for index, value in self.y_test.items():
            if value > limit:
                removed_x_indexes.append(count)
                removed_y_indexes.append(index)
            count += 1
        # remove
        self.x_test_std = np.delete(self.x_test_std, removed_x_indexes, axis=0)
        self.y_test.drop(removed_y_indexes, inplace=True)

    def predict(self, limit: Optional[int] = None) -> np.ndarray:
        if self.predicted is None:
            if hasattr(self, "_create_regression_model"):
                regressor = self._create_regression_model()
                if limit is not None:
                    self._grant_upper_limit(limit)
                self.predicted = regressor.predict(self.x_test_std)
            else:
                raise ValueError("Prediction method has not been implemented yet.")
        return self.predicted

    def evaluate(self, limit: Optional[int] = None) -> MLResult:
        predicted = self.predict(limit)
        r2 = r2_score(self.y_test, predicted)
        mae = mean_absolute_error(self.y_test, predicted)
        return MLResult(r2_score=r2, mae=mae)

    def plt_show(self):
        predicted = self.predict()
        plt.xlabel("predicted")
        plt.ylabel("actual")
        plt.scatter(predicted, self.y_test)
        plt.show()


class LinearRegressionPrediction(BasePrediction):
    def _create_regression_model(self) -> LinearRegression:
        self._preprocess()
        linear_regression = LinearRegression()
        linear_regression.fit(self.x_train_std, self.y_train)
        return linear_regression


class RandomForestPrediction(BasePrediction):
    def _create_regression_model(self) -> RandomForestRegressor:
        self._preprocess()
        random_forest_regressor = RandomForestRegressor()
        random_forest_regressor.fit(self.x_train_std, self.y_train)
        return random_forest_regressor


class GradientBoostingPrediction(BasePrediction):
    def _create_regression_model(self) -> GradientBoostingRegressor:
        self._preprocess()
        regressor = GradientBoostingRegressor()
        regressor.fit(self.x_train_std, self.y_train)
        return regressor


class DecisionTreePrediction(BasePrediction):
    def _create_regression_model(self) -> DecisionTreeRegressor:
        self._preprocess()
        regressor = DecisionTreeRegressor()
        regressor.fit(self.x_train_std, self.y_train)
        return regressor


class LogisticRegressionPrediction(BasePrediction):
    def _create_regression_model(self) -> LogisticRegression:
        self._preprocess()
        logistic_regression = LogisticRegression(max_iter=10**4)
        logistic_regression.fit(self.x_train_std, self.y_train)
        return logistic_regression
