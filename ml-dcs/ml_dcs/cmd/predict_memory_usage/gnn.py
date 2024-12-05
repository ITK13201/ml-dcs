import argparse
import datetime
import os.path
from logging import getLogger

from ml_dcs.cmd.base import BaseCommand
from ml_dcs.internal.signal.signal import SignalUtil
from ml_dcs.usecases.gnn_evaluator import GNNEvaluator

logger = getLogger(__name__)


class PredictMemoryUsageGNNCommand(BaseCommand):
    name = "gnn"
    help = "Predict memory usage using GNN"

    def add_arguments(self, parser: argparse.ArgumentParser):
        # input
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
            "-e",
            "--max-epochs",
            type=str,
            required=True,
            help="Maximum number of epochs",
        )
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
        self.max_epochs = None
        self.signal_dir = None
        self.threshold = None
        # === parameters ===
        self.target_name = "memory_usage"
        self.output_dir_path = None
        self.training_result_output_file_path = None
        self.testing_result_output_file_path = None
        self.lts_gnn_model_output_file_path = None
        self.regression_model_output_file_path = None

    def execute(self, args: argparse.Namespace):
        logger.info("PredictMemoryUsageGNNCommand started")

        # args
        self.input_dir_path: str = args.input_dir_path
        self.bench_result_file_path: str = args.bench_result_file
        self.output_base_dir_path: str = args.output_base_dir_path
        self.max_epochs: int = int(args.max_epochs)
        self.signal_dir: str | None = args.signal_dir
        self.threshold: float = args.threshold

        # build output dir
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir_path: str = os.path.join(self.output_base_dir_path, now_str)
        os.makedirs(self.output_dir_path, exist_ok=True)

        # signal dir
        if self.signal_dir is not None:
            SignalUtil.set_signal_dir(self.signal_dir)

        # output file paths
        self.training_result_output_file_path: str = os.path.join(
            self.output_dir_path, "training-result.json"
        )
        self.testing_result_output_file_path: str = os.path.join(
            self.output_dir_path, "testing-result.json"
        )
        self.lts_gnn_model_output_file_path: str = os.path.join(
            self.output_dir_path, "lts-gnn-model.pth"
        )
        self.regression_model_output_file_path: str = os.path.join(
            self.output_dir_path, "regression-model.pth"
        )

        evaluator = GNNEvaluator(
            input_dir_path=self.input_dir_path,
            training_result_output_file_path=self.training_result_output_file_path,
            testing_result_output_file_path=self.testing_result_output_file_path,
            lts_gnn_model_output_file_path=self.lts_gnn_model_output_file_path,
            regression_model_output_file_path=self.regression_model_output_file_path,
            max_epochs=self.max_epochs,
            target_name=self.target_name,
            threshold=self.threshold,
        )
        evaluator.evaluate()

        logger.info("PredictMemoryUsageGNNCommand finished")
