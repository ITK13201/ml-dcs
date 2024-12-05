import argparse
from typing import Type

from ml_dcs.cmd.base import BaseCommand
from ml_dcs.cmd.check_gpu import CheckGPUCommand
from ml_dcs.cmd.predict_calculation_time.gnn import PredictCalculationTimeGNNCommand
from ml_dcs.cmd.predict_calculation_time.root import PredictCalculationTimeCommand
from ml_dcs.cmd.predict_calculation_time.simple import (
    PredictCalculationTimeSimpleCommand,
)
from ml_dcs.cmd.predict_memory_usage.gnn import PredictMemoryUsageGNNCommand
from ml_dcs.cmd.predict_memory_usage.root import PredictMemoryUsageCommand
from ml_dcs.cmd.predict_memory_usage.simple import PredictMemoryUsageSimpleCommand
from ml_dcs.cmd.prepare_dataset import PrepareDatasetCommand
from ml_dcs.cmd.try_sample import TrySampleCommand


class RootCommand:
    description = "Machine Learning for Discrete Controller Synthesis"
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command")

    def add_command(
        self, command_class: Type[BaseCommand], parent=None, is_parent: bool = False
    ):
        if parent is None:
            parent = self.subparsers
        command = command_class()
        parser = parent.add_parser(command.name, help=command.help)
        command.add_arguments(parser)
        if not is_parent:
            parser.set_defaults(handler=command.execute)
        else:
            subparser = parser.add_subparsers(title=command.name, dest=command.name)
            return subparser

    def __init__(self):
        self.add_command(TrySampleCommand)
        predict_calculation_time_command = self.add_command(
            PredictCalculationTimeCommand, is_parent=True
        )
        self.add_command(
            PredictCalculationTimeSimpleCommand, parent=predict_calculation_time_command
        )
        self.add_command(
            PredictCalculationTimeGNNCommand, parent=predict_calculation_time_command
        )
        predict_memory_usage_command = self.add_command(
            PredictMemoryUsageCommand, is_parent=True
        )
        self.add_command(
            PredictMemoryUsageSimpleCommand, parent=predict_memory_usage_command
        )
        self.add_command(
            PredictMemoryUsageGNNCommand, parent=predict_memory_usage_command
        )
        self.add_command(CheckGPUCommand)
        self.add_command(PrepareDatasetCommand)

    def execute(self):
        args = self.parser.parse_args()
        if hasattr(args, "handler"):
            args.handler(args)
        else:
            if args.command is None:
                self.parser.print_help()
            else:
                self.subparsers.choices[args.command].print_help()
