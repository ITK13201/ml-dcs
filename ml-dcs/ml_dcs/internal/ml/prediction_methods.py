from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_dcs.domain.ml import MLResult


class BasePrediction:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.x_train: Optional[pd.DataFrame] = None
        self.x_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        self.x_train_std: Optional[pd.DataFrame] = None
        self.x_test_std: Optional[pd.DataFrame] = None
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

    def predict(self) -> np.ndarray:
        if self.predicted is None:
            if hasattr(self, "_create_regression_model"):
                regressor = self._create_regression_model()
                self.predicted = regressor.predict(self.x_test_std)
            else:
                raise ValueError("Prediction method has not been implemented yet.")
        return self.predicted

    def evaluate(self) -> MLResult:
        predicted = self.predict()
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
