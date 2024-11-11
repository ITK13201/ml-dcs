import json
import logging
import os
from datetime import datetime

from ml_dcs.domain.ml import MLMemoryUsagePredictionInput2
from ml_dcs.internal.ml.prediction_methods import (
    DecisionTreePrediction,
    GradientBoostingPrediction,
    LogisticRegressionPrediction,
    RandomForestPrediction,
)
from ml_dcs.usecases.evaluator import RandomStateEvaluator

logger = logging.getLogger(__name__)


class PredictMemoryUsageSimpleExecutor:
    def __init__(self, input_dir: str, output_dir: str, bench_result_file_path: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.bench_result_file_path = bench_result_file_path
        self.mode = "simple"

    def execute(self):
        logger.info("mode: %s", self.mode)

        # GBDT
        logger.info("GBDT started")
        self._predict_using_gbdt(self.input_dir, self.output_dir)
        logger.info("GBDT finished")

        # RF
        logger.info("RF started")
        self._predict_using_rf(self.input_dir, self.output_dir)
        logger.info("RF finished")

        # DT
        logger.info("DT started")
        self._predict_using_dt(self.input_dir, self.output_dir)
        logger.info("DT finished")

        # LR
        logger.info("LR started")
        self._predict_using_lr(self.input_dir, self.output_dir)
        logger.info("LR finished")

    def _get_output_file_name(self, ml_algorithm: str) -> str:
        return (
            "_".join(
                [
                    "memory-usage",
                    self.mode,
                    ml_algorithm,
                    datetime.now().strftime("%y%m%d%H%M%S"),
                ]
            )
            + ".json"
        )

    def _predict_using_gbdt(self, input_dir: str, output_dir: str):
        evaluator = RandomStateEvaluator(
            input_dir_path=input_dir,
            ml_input_class=MLMemoryUsagePredictionInput2,
            prediction_class=GradientBoostingPrediction,
        )
        result = evaluator.evaluate()
        output_name = self._get_output_file_name(ml_algorithm="GBDT")
        output_path = os.path.join(output_dir, output_name)
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    def _predict_using_rf(self, input_dir: str, output_dir: str):
        evaluator = RandomStateEvaluator(
            input_dir_path=input_dir,
            ml_input_class=MLMemoryUsagePredictionInput2,
            prediction_class=RandomForestPrediction,
        )
        result = evaluator.evaluate()
        output_file_name = self._get_output_file_name(ml_algorithm="RF")
        output_path = os.path.join(output_dir, output_file_name)
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    def _predict_using_dt(self, input_dir: str, output_dir: str):
        evaluator = RandomStateEvaluator(
            input_dir_path=input_dir,
            ml_input_class=MLMemoryUsagePredictionInput2,
            prediction_class=DecisionTreePrediction,
        )
        result = evaluator.evaluate()
        output_name = self._get_output_file_name(ml_algorithm="DT")
        output_path = os.path.join(output_dir, output_name)
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    def _predict_using_lr(self, input_dir: str, output_dir: str):
        evaluator = RandomStateEvaluator(
            input_dir_path=input_dir,
            ml_input_class=MLMemoryUsagePredictionInput2,
            prediction_class=LogisticRegressionPrediction,
        )
        result = evaluator.evaluate()
        output_name = self._get_output_file_name(ml_algorithm="LR")
        output_path = os.path.join(output_dir, output_name)
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
