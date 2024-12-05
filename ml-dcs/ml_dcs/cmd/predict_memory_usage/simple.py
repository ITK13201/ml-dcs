import argparse
import datetime
import logging
import os

from ml_dcs.cmd.base import BaseCommand
from ml_dcs.domain.evaluation import EvaluationTarget
from ml_dcs.internal.ml.simple import RegressionAlgorithm
from ml_dcs.internal.signal.signal import SignalUtil
from ml_dcs.usecases.simple_evaluator import MLSimpleEvaluator

logger = logging.getLogger(__name__)


class PredictMemoryUsageSimpleCommand(BaseCommand):
    name = "simple"
    help = "Predict memory usage using simple ML methods"

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-i",
            "--input-dir-path",
            type=str,
            required=True,
            help="Input data directory",
        )
        parser.add_argument(
            "-f",
            "--bench-result-file",
            type=str,
            required=True,
            help="Bench result file path",
        )
        # output
        parser.add_argument(
            "-o",
            "--output-base-dir-path",
            type=str,
            required=True,
            help="Output base directory path",
        )
        # additional
        parser.add_argument(
            "-s",
            "--signal-dir",
            type=str,
            required=False,
            help="Signal directory path",
        )
        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            required=False,
            help="Threshold for memory usage",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # === args ===
        self.input_dir_path = None
        self.bench_result_file_path = None
        self.output_base_dir_path = None
        self.signal_dir = None
        self.threshold = None
        # === parameters ===
        self.output_dir_path = None
        self.target = EvaluationTarget.MEMORY_USAGE

    def execute(self, args: argparse.Namespace):
        logger.info("PredictMemoryUsageSimpleCommand started")

        # args
        self.input_dir_path: str = args.input_dir_path
        self.bench_result_file_path: str = args.bench_result_file
        self.output_base_dir_path: str = args.output_base_dir_path
        self.signal_dir: str | None = args.signal_dir
        self.threshold: float = args.threshold

        # build output dir
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir_path: str = os.path.join(self.output_base_dir_path, now_str)
        os.makedirs(self.output_dir_path, exist_ok=True)

        # signal dir
        if self.signal_dir is not None:
            SignalUtil.set_signal_dir(self.signal_dir)

        # GBDT
        logger.info("GBDT started")
        self._evaluate_gbdt()
        logger.info("GBDT finished")

        # RF
        logger.info("RF started")
        self._evaluate_rf()
        logger.info("RF finished")

        # DT
        logger.info("DT started")
        self._evaluate_dt()
        logger.info("DT finished")

        # LR
        logger.info("LR started")
        self._evaluate_lr()
        logger.info("LR finished")

        logger.info("PredictMemoryUsageSimpleCommand finished")

    def _get_training_result_output_file_path(self, algorithm: str) -> str:
        return os.path.join(self.output_dir_path, f"training-result_{algorithm}.json")

    def _get_testing_result_output_file_path(self, algorithm: str) -> str:
        return os.path.join(self.output_dir_path, f"testing-result_{algorithm}.json")

    def _evaluate_gbdt(self):
        evaluator = MLSimpleEvaluator(
            input_dir_path=self.input_dir_path,
            training_result_output_file_path=self._get_training_result_output_file_path(
                "GBDT"
            ),
            testing_result_output_file_path=self._get_testing_result_output_file_path(
                "GBDT"
            ),
            algorithm=RegressionAlgorithm.GRADIENT_BOOSTING_DECISION_TREE,
            target=self.target,
            threshold=self.threshold,
        )
        evaluator.evaluate()

    def _evaluate_rf(self):
        evaluator = MLSimpleEvaluator(
            input_dir_path=self.input_dir_path,
            training_result_output_file_path=self._get_training_result_output_file_path(
                "RF"
            ),
            testing_result_output_file_path=self._get_testing_result_output_file_path(
                "RF"
            ),
            algorithm=RegressionAlgorithm.RANDOM_FOREST,
            target=self.target,
            threshold=self.threshold,
        )
        evaluator.evaluate()

    def _evaluate_dt(self):
        evaluator = MLSimpleEvaluator(
            input_dir_path=self.input_dir_path,
            training_result_output_file_path=self._get_training_result_output_file_path(
                "DT"
            ),
            testing_result_output_file_path=self._get_testing_result_output_file_path(
                "DT"
            ),
            algorithm=RegressionAlgorithm.DECISION_TREE,
            target=self.target,
            threshold=self.threshold,
        )
        evaluator.evaluate()

    def _evaluate_lr(self):
        evaluator = MLSimpleEvaluator(
            input_dir_path=self.input_dir_path,
            training_result_output_file_path=self._get_training_result_output_file_path(
                "LR"
            ),
            testing_result_output_file_path=self._get_testing_result_output_file_path(
                "LR"
            ),
            algorithm=RegressionAlgorithm.LOGISTIC_REGRESSION,
            target=self.target,
            threshold=self.threshold,
        )
        evaluator.evaluate()
