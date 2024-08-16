import argparse

from ml_dcs.cmd.predict_calculation_time import PredictCalculationTimeCommand
from ml_dcs.cmd.predict_memory_usage import PredictMemoryUsageCommand
from ml_dcs.cmd.try_sample import TrySampleCommand


class RootCommand:
    description = "Machine Learning for Discrete Controller Synthesis"

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=RootCommand.description)
        subparsers = self.parser.add_subparsers()

        # try_sample
        try_sample_command = TrySampleCommand()
        parser_try_sample = subparsers.add_parser(
            try_sample_command.name, help=try_sample_command.help
        )
        try_sample_command.add_arguments(parser_try_sample)
        parser_try_sample.set_defaults(handler=try_sample_command.execute)

        # predict_calculation_time
        predict_calculation_time_command = PredictCalculationTimeCommand()
        parser_predict_calculation_time = subparsers.add_parser(
            predict_calculation_time_command.name,
            help=predict_calculation_time_command.help,
        )
        predict_calculation_time_command.add_arguments(parser_predict_calculation_time)
        parser_predict_calculation_time.set_defaults(
            handler=predict_calculation_time_command.execute
        )

        # predict_memory_usage
        predict_memory_usage_command = PredictMemoryUsageCommand()
        parser_predict_memory_usage = subparsers.add_parser(
            predict_memory_usage_command.name,
            help=predict_memory_usage_command.help,
        )
        predict_calculation_time_command.add_arguments(parser_predict_memory_usage)
        parser_predict_memory_usage.set_defaults(
            handler=predict_memory_usage_command.execute
        )

    def execute(self):
        args = self.parser.parse_args()
        if hasattr(args, "handler"):
            args.handler(args)
        else:
            self.parser.print_help()
