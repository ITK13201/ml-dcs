import operator
import time
from statistics import variance
from typing import Tuple, List

import pandas as pd
from matplotlib import pyplot as plt

from ml_dcs.domain.evaluation import (
    EvaluationResult,
    EvaluationResultBestAccuracyScores,
)
from ml_dcs.domain.ml import MLResult
from ml_dcs.internal.mtsa.data_utils import MTSADataUtil


class RandomStateEvaluator:
    random_states = [i for i in range(0, 3000, 1)]

    def __init__(self, input_dir_path: str, ml_input_class, prediction_class):
        self.input_dir_path = input_dir_path
        self.ml_input_class = ml_input_class
        self.prediction_class = prediction_class
        self.r2_scores = None
        self.mae_scores = None
        # minutes (min)
        self.prediction_time = None
        self.r2_score_variance = None
        self.best_r2_score = None
        self.best_mae_score = None
        self.best_random_state = None

    def _execute_with_single_random_state(
        self, df: pd.DataFrame, random_state: int
    ) -> MLResult:
        prediction = self.prediction_class(df, random_state)
        result = prediction.evaluate()
        return result

    def _execute_with_multiple_random_states(
        self,
    ) -> Tuple[List[float], List[float]]:
        util = MTSADataUtil(input_dir_path=self.input_dir_path)
        df = util.get_dataframe(ml_input_class=self.ml_input_class)

        start_time = time.perf_counter()

        r2_scores = []
        mae_sores = []
        for random_state in self.random_states:
            result = self._execute_with_single_random_state(
                df=df, random_state=random_state
            )
            r2_scores.append(result.r2_score)
            mae_sores.append(result.mae)

        end_time = time.perf_counter()
        self.prediction_time = end_time - start_time

        return r2_scores, mae_sores

    def evaluate(self) -> EvaluationResult:
        if self.r2_scores is None or self.mae_scores is None:
            (
                self.r2_scores,
                self.mae_scores,
            ) = self._execute_with_multiple_random_states()
        self.best_random_state, self.best_r2_score = max(
            enumerate(self.r2_scores), key=operator.itemgetter(1)
        )
        self.r2_score_variance = variance(self.r2_scores)
        self.best_mae_score = self.mae_scores[self.best_random_state]

        result = EvaluationResult(
            ml_algorithm=self.prediction_class.ml_algorithm,
            ml_input_class=self.ml_input_class.__name__,
            random_states=self.random_states,
            r2_scores=self.r2_scores,
            mae_scores=self.mae_scores,
            r2_score_variance=self.r2_score_variance,
            prediction_time=self.prediction_time,
            best_accuracy_scores=EvaluationResultBestAccuracyScores(
                random_state=self.best_random_state,
                r2_score=self.best_r2_score,
                mae_score=self.best_mae_score,
            ),
        )
        return result

    def plt_show(self):
        if self.r2_scores is None or self.mae_scores is None:
            (
                self.r2_scores,
                self.mae_scores,
            ) = self._execute_with_multiple_random_states()
        plt.xlabel("The value of seed")
        plt.ylabel("Coefficient of determination $R^2$")
        plt.xlim(0, 3000)
        plt.ylim(-0.5, 1)
        plt.plot([0, 3000], [0, 0], color="black", linestyle="--")
        plt.scatter(
            self.random_states,
            self.r2_scores,
            c="white",
            edgecolors="blue",
            s=10,
        )
        plt.show()

        # plt.xlabel("The value of seed")
        # plt.ylabel("Mean absolute error")
        # plt.scatter(self.random_states, self.mae_scores)
        # plt.show()

    def get_graph_elements(self) -> Tuple[List[float], List[float]]:
        if self.r2_scores is None or self.mae_scores is None:
            (
                self.r2_scores,
                self.mae_scores,
            ) = self._execute_with_multiple_random_states()
        return self.r2_scores, self.mae_scores
