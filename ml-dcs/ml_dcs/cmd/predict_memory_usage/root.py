import argparse
import logging

from ml_dcs.cmd.base import BaseCommand

logger = logging.getLogger(__name__)


class PredictMemoryUsageCommand(BaseCommand):
    name = "predict_memory_usage"
    help = "Predict memory usage"

    def add_arguments(self, parser: argparse.ArgumentParser):
        pass

    def execute(self, args: argparse.Namespace):
        pass
