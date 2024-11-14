import argparse
from typing import Type

from ml_dcs.cmd.base import BaseCommand
from ml_dcs.cmd.check_gpu import CheckGPUCommand
from ml_dcs.cmd.predict_calculation_time import PredictCalculationTimeCommand
from ml_dcs.cmd.predict_memory_usage import PredictMemoryUsageCommand
from ml_dcs.cmd.try_sample import TrySampleCommand


class RootCommand:
    description = "Machine Learning for Discrete Controller Synthesis"
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command")

    def add_command(self, command_class: Type[BaseCommand]):
        command = command_class()
        subparser = self.subparsers.add_parser(command.name, help=command.help)
        command.add_arguments(subparser)
        subparser.set_defaults(handler=command.execute)

    def __init__(self):
        self.add_command(TrySampleCommand)
        self.add_command(PredictCalculationTimeCommand)
        self.add_command(PredictMemoryUsageCommand)
        self.add_command(CheckGPUCommand)

    def execute(self):
        args = self.parser.parse_args()
        if hasattr(args, "handler"):
            args.handler(args)
        else:
            self.parser.print_help()
