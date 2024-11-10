import argparse
import logging

from ml_dcs.cmd.base import BaseCommand
from ml_dcs.executors.predict_memory_usage.gnn import PredictMemoryUsageGNNExecutor
from ml_dcs.executors.predict_memory_usage.simple import (
    PredictMemoryUsageSimpleExecutor,
)

logger = logging.getLogger(__name__)


class PredictMemoryUsageCommand(BaseCommand):
    name = "predict_memory_usage"
    help = "Predict memory usage"

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
        logger.info("memory usage prediction started")

        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.mode = args.mode

        match self.mode:
            case "simple":
                PredictMemoryUsageSimpleExecutor(
                    self.input_dir, self.output_dir
                ).execute()
            case "gnn":
                PredictMemoryUsageGNNExecutor(self.input_dir, self.output_dir).execute()
            case _:
                raise RuntimeError("Invalid mode")

        logger.info("memory usage prediction finished")
