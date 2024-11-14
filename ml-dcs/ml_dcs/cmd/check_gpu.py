import argparse
import logging

import torch

from ml_dcs.cmd.base import BaseCommand

logger = logging.getLogger(__name__)


class CheckGPUCommand(BaseCommand):
    name = "check_gpu"
    help = "Check GPU"

    def add_arguments(self, parser: argparse.ArgumentParser):
        pass

    def execute(self, args: argparse.Namespace):
        if torch.cuda.is_available():
            logger.info("GPU is available")
        else:
            logger.warning("GPU is not available")
