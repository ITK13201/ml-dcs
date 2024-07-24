from typing import Optional

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


class LinearRegressionPrediction(BasePrediction):
    def _create_regression_model(self) -> LinearRegression:
        self._preprocess()
        linear_regression = LinearRegression()
        linear_regression.fit(self.x_train_std, self.y_train)
        return linear_regression

    def predict(self) -> MLResult:
        linear_regression = self._create_regression_model()
        predicted: np.ndarray = linear_regression.predict(self.x_test_std)

        r2 = r2_score(self.y_test, predicted)
        mae = mean_absolute_error(self.y_test, predicted)
        return MLResult(r2_score=r2, mae=mae)


class RandomForestPrediction(BasePrediction):
    def _create_regression_model(self) -> RandomForestRegressor:
        self._preprocess()
        random_forest_regressor = RandomForestRegressor()
        random_forest_regressor.fit(self.x_train_std, self.y_train)
        return random_forest_regressor

    def predict(self) -> MLResult:
        random_forest_regressor = self._create_regression_model()
        predicted: np.ndarray = random_forest_regressor.predict(self.x_test_std)

        r2 = r2_score(self.y_test, predicted)
        mae = mean_absolute_error(self.y_test, predicted)
        return MLResult(r2_score=r2, mae=mae)


class GradientBoostingPrediction(BasePrediction):
    def _create_regression_model(self) -> GradientBoostingRegressor:
        self._preprocess()
        regressor = GradientBoostingRegressor()
        regressor.fit(self.x_train_std, self.y_train)
        return regressor

    def predict(self) -> MLResult:
        regressor = self._create_regression_model()
        predicted: np.ndarray = regressor.predict(self.x_test_std)

        r2 = r2_score(self.y_test, predicted)
        mae = mean_absolute_error(self.y_test, predicted)
        return MLResult(r2_score=r2, mae=mae)
