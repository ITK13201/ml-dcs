import argparse
import logging

from ml_dcs.cmd.base import BaseCommand
from ml_dcs.executors.predict_calculation_time.gnn import (
    PredictCalculationTimeGNNExecutor,
)
from ml_dcs.executors.predict_calculation_time.simple import (
    PredictCalculationTimeSimpleExecutor,
)

logger = logging.getLogger(__name__)


class PredictCalculationTimeCommand(BaseCommand):
    name = "predict_calculation_time"
    help = "Predict calculation time"

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-i", "--input-dir", type=str, required=True, help="Input data directory"
        )
        parser.add_argument(
            "-o", "--output-dir", type=str, required=True, help="Output data directory"
        )
        parser.add_argument(
            "-m", "--mode", type=str, required=True, help="Prediction mode (simple/gnn)"
        )

    def __init__(self):
        super().__init__()
        self.input_dir = ""
        self.output_dir = ""
        self.mode = ""

    def execute(self, args: argparse.Namespace):
        logger.info("calculation time prediction started")

        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.mode = args.mode

        match self.mode:
            case "simple":
                PredictCalculationTimeSimpleExecutor(
                    self.input_dir, self.output_dir
                ).execute()
            case "gnn":
                PredictCalculationTimeGNNExecutor(
                    self.input_dir, self.output_dir
                ).execute()
            case _:
                raise RuntimeError("Invalid mode")

        logger.info("calculation time prediction finished")
