import argparse


class PredictCalculationTimeCommand:
    name = 'predict_calculation_time'
    help = "Predict calculation time"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument("-i", "--input-dir", type=str, required=True, help="Input data directory")
        parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output data directory")

    @classmethod
    def execute(cls, args: argparse.Namespace):
        print(args)
