import argparse
import json
import logging
import os
from datetime import datetime

from ml_dcs.domain.ml import MLMemoryUsagePredictionInput2
from ml_dcs.internal.ml.prediction_methods import (
    GradientBoostingPrediction,
    RandomForestPrediction,
    DecisionTreePrediction,
    LogisticRegressionPrediction,
)
from ml_dcs.usecase.evaluator import RandomStateEvaluator

logger = logging.getLogger(__name__)


class PredictMemoryUsageCommand:
    name = "predict_memory_usage"
    help = "Predict memory usage"

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-i", "--input-dir", type=str, required=True, help="Input data directory"
        )
        parser.add_argument(
            "-o", "--output-dir", type=str, required=True, help="Output data directory"
        )

    def execute(self, args: argparse.Namespace):
        logger.info("memory usage prediction started")

        # GBDT
        logger.info("GBDT started")
        self._predict_using_gbdt(args.input_dir, args.output_dir)
        logger.info("GBDT finished")

        # RF
        logger.info("RF started")
        self._predict_using_rf(args.input_dir, args.output_dir)
        logger.info("RF finished")

        # DT
        logger.info("DT started")
        self._predict_using_dt(args.input_dir, args.output_dir)
        logger.info("DT finished")

        # LR
        logger.info("LR started")
        self._predict_using_lr(args.input_dir, args.output_dir)
        logger.info("LR finished")

        logger.info("memory usage prediction finished")

    @staticmethod
    def _get_output_file_name(ml_algorithm: str) -> str:
        return (
            "_".join(
                [
                    datetime.now().strftime("%y%m%d%H%M%S"),
                    "memory-usage",
                    ml_algorithm,
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
