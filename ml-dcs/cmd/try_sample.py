import argparse


class TrySampleCommand:
    name = 'try_sample'
    help = "Try sample of scikit-learn dataset"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    def execute(cls, args: argparse.Namespace):
        print(args)
