import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from ml_dcs.cmd.base import BaseCommand


class TrySampleCommand(BaseCommand):
    name = "try_sample"
    help = "Try sample of scikit-learn dataset"

    def add_arguments(self, parser: argparse.ArgumentParser):
        pass

    def execute(self, args: argparse.Namespace):
        dataset: Bunch = fetch_california_housing()

        # create dataframe
        df: pd.DataFrame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df["HousePrice"] = dataset.target

        # create train and test data
        x_train, x_test, y_train, y_test = train_test_split(
            df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42
        )

        # standardize
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_std: pd.DataFrame = scaler.transform(x_train)
        x_test_std: pd.DataFrame = scaler.transform(x_test)

        # predict with linear regression
        linear_regression = LinearRegression()
        linear_regression.fit(x_train_std, y_train)

        predictions: np.ndarray = linear_regression.predict(x_test_std)
        r2 = r2_score(y_test, predictions)
        print(f"R2: {r2:.2f}")
        mae = mean_absolute_error(y_test, predictions)
        print(f"MAE: {mae:.2f}")
        print(f"Coefficients: {linear_regression.coef_}")
        print(f"Intercept: {linear_regression.intercept_}")
