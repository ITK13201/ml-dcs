from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml_dcs.domain.ml import MLResult
from ml_dcs.internal.graph.graph import FixedOrderFormatter

DEFAULT_RANDOM_STATE = 42


class BasePrediction:
    ml_algorithm = None

    def __init__(self, df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE):
        self.df = df
        self.x_train: Optional[pd.DataFrame] = None
        self.x_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.x_train_std: Optional[pd.DataFrame] = None
        self.x_test_std: Optional[np.ndarray] = None
        self.predicted: Optional[np.ndarray] = None
        self.random_state = random_state

    def _preprocess(self):
        # create train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df.iloc[:, :-1],
            self.df.iloc[:, -1],
            test_size=0.2,
            random_state=DEFAULT_RANDOM_STATE,
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

    def plt_show(
        self,
        xlim: Optional[Tuple[float]] = None,
        ylim: Optional[Tuple[float]] = None,
        unit: Optional[str] = None,
    ):
        predicted = self.predict()

        if unit is None:
            plt.xlabel("Predicted values")
        else:
            plt.xlabel("Predicted values ({})".format(unit))

        if unit is None:
            plt.ylabel("Actual values")
        else:
            plt.ylabel("Actual values ({})".format(unit))

        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FixedOrderFormatter(order_of_mag=6))
        ax.yaxis.set_major_formatter(FixedOrderFormatter(order_of_mag=6))
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

        ax.scatter(
            predicted,
            self.y_test,
            c="white",
            edgecolors="blue",
            s=50,
        )
        plt.show()


class LinearRegressionPrediction(BasePrediction):
    ml_algorithm = "LinearRegression"

    def _create_regression_model(self) -> LinearRegression:
        self._preprocess()
        linear_regression = LinearRegression()
        linear_regression.fit(self.x_train_std, self.y_train)
        return linear_regression


class RandomForestPrediction(BasePrediction):
    ml_algorithm = "RandomForest"

    def _create_regression_model(self) -> RandomForestRegressor:
        self._preprocess()
        random_forest_regressor = RandomForestRegressor(random_state=self.random_state)
        random_forest_regressor.fit(self.x_train_std, self.y_train)
        return random_forest_regressor


class GradientBoostingPrediction(BasePrediction):
    ml_algorithm = "GradientBoostingDecisionTree"

    def _create_regression_model(self) -> GradientBoostingRegressor:
        self._preprocess()
        regressor = GradientBoostingRegressor(random_state=self.random_state)
        regressor.fit(self.x_train_std, self.y_train)
        return regressor


class DecisionTreePrediction(BasePrediction):
    ml_algorithm = "DecisionTree"

    def _create_regression_model(self) -> DecisionTreeRegressor:
        self._preprocess()
        regressor = DecisionTreeRegressor(random_state=self.random_state)
        regressor.fit(self.x_train_std, self.y_train)
        return regressor


class LogisticRegressionPrediction(BasePrediction):
    ml_algorithm = "LogisticRegression"

    def _create_regression_model(self) -> LogisticRegression:
        self._preprocess()
        logistic_regression = LogisticRegression(
            random_state=self.random_state, max_iter=10**4
        )
        logistic_regression.fit(self.x_train_std, self.y_train)
        return logistic_regression
