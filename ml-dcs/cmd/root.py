import argparse

from .try_sample import TrySampleCommand


class RootCommand:
    description = "Machine Learning for Discrete Controller Synthesis"

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=RootCommand.description)
        subparsers = self.parser.add_subparsers()

        parser_try_sample = subparsers.add_parser(TrySampleCommand.name, help=TrySampleCommand.help)
        TrySampleCommand.add_arguments(parser_try_sample)
        parser_try_sample.set_defaults(handler=TrySampleCommand.execute)

    def execute(self):
        args = self.parser.parse_args()
        if hasattr(args, 'handler'):
            args.handler(args)
        else:
            self.parser.print_help()
