import datetime
import json
from logging import getLogger
from typing import Type

import numpy as np
import pandas as pd

from ml_dcs.domain.ml_simple import (
    BaseMLInput,
    MLSimpleTestingResult,
    MLSimpleTestingResultFinal,
    MLSimpleTestingResultSet,
    MLSimpleTrainingResult,
    MLSimpleTrainingResultSet,
)
from ml_dcs.internal.ml.simple import (
    MLSimpleDataUtil,
    RegressionAlgorithm,
    RegressionModel,
)
from ml_dcs.internal.mtsa.data_utils import MTSADataUtil

logger = getLogger(__name__)


class MLSimpleEvaluator:
    RANDOM_STATES = [i for i in range(0, 3000, 10)]

    def __init__(
        self,
        input_dir_path: str,
        training_result_output_file_path: str,
        testing_result_output_file_path: str,
        algorithm: RegressionAlgorithm,
        ml_input_class: Type[BaseMLInput],
    ):
        self.input_dir_path = input_dir_path
        self.training_result_output_file_path = training_result_output_file_path
        self.testing_result_output_file_path = testing_result_output_file_path
        self.algorithm = algorithm
        self.ml_input_class = ml_input_class

    def train(
        self, model: RegressionModel, x_train_std: np.ndarray, y_train: pd.Series
    ) -> MLSimpleTrainingResult:
        started_at = datetime.datetime.now()

        model.train(x_train_std, y_train)

        finished_at = datetime.datetime.now()
        result = MLSimpleTrainingResult(
            algorithm=self.algorithm.value,
            random_state=model.random_state,
            started_at=started_at,
            finished_at=finished_at,
        )
        return result

    def test(
        self, model: RegressionModel, x_test_std: np.ndarray, y_test: pd.Series
    ) -> MLSimpleTestingResult:
        started_at = datetime.datetime.now()

        predicted_values = model.predict(x_test_std)

        finished_at = datetime.datetime.now()
        result = MLSimpleTestingResult(
            algorithm=self.algorithm.value,
            random_state=model.random_state,
            actual_values=y_test.values.tolist(),
            predicted_values=predicted_values.tolist(),
            started_at=started_at,
            finished_at=finished_at,
        )
        return result

    def evaluate(
        self,
    ):
        # load data
        logger.info("loading data...")
        mtsa_data_util = MTSADataUtil(input_dir_path=self.input_dir_path)
        df = mtsa_data_util.get_dataframe(ml_input_class=self.ml_input_class)

        ml_data_util = MLSimpleDataUtil(df)
        x_train_std, x_test_std, y_train, y_test = ml_data_util.get_dataset()

        # evaluate
        logger.info("evaluation started")

        train_results = []
        test_result_at_best_accuracy = None
        test_result_finals = []
        random_states_length = len(self.RANDOM_STATES)
        for index, random_state in enumerate(self.RANDOM_STATES):
            model = RegressionModel(algorithm=self.algorithm, random_state=random_state)
            train_result = self.train(model, x_train_std, y_train)
            test_result = self.test(model, x_test_std, y_test)

            train_results.append(train_result)
            if test_result_at_best_accuracy is None:
                test_result_at_best_accuracy = test_result
            elif test_result_at_best_accuracy.r_squared < test_result.r_squared:
                test_result_at_best_accuracy = test_result
            test_result_final = MLSimpleTestingResultFinal(**test_result.model_dump())
            test_result_finals.append(test_result_final)

            if (index + 1) % 10 == 0:
                logger.info(
                    "evaluating... : %s",
                    json.dumps(
                        {
                            "progress": f"{index+1}/{random_states_length}",
                            "best_accuracy": test_result_at_best_accuracy.r_squared,
                        }
                    ),
                )

        # result
        train_result_set = MLSimpleTrainingResultSet(
            algorithm=self.algorithm.value,
            results=train_results,
        )
        with open(
            self.training_result_output_file_path, mode="w", encoding="utf-8"
        ) as f:
            f.write(train_result_set.model_dump_json(indent=2, by_alias=True))
        test_result_set = MLSimpleTestingResultSet(
            algorithm=self.algorithm.value,
            results=test_result_finals,
            result_at_best_accuracy=test_result_at_best_accuracy,
        )
        with open(
            self.testing_result_output_file_path, mode="w", encoding="utf-8"
        ) as f:
            f.write(test_result_set.model_dump_json(indent=2, by_alias=True))

        logger.info("evaluation finished")
