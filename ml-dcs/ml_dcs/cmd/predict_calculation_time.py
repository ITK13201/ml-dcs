import argparse

from ml_dcs.internal.mtsa.data_utils import MTSADataUtil


class PredictCalculationTimeCommand:
    name = "predict_calculation_time"
    help = "Predict calculation time"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-i", "--input-dir", type=str, required=True, help="Input data directory"
        )
        parser.add_argument(
            "-o", "--output-dir", type=str, required=True, help="Output data directory"
        )

    @classmethod
    def execute(cls, args: argparse.Namespace):
        mtsa_util = MTSADataUtil(input_dir_path="./tmp/2024-07-04/input")
        data = mtsa_util._parse_data()

        for mtsa_result in data:
            print(mtsa_result.lts, mtsa_result.duration_iso)
