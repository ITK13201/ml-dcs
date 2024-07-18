import argparse


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
        pass
        # mtsa_util = MTSADataUtil(input_dir_path="./tmp/input/2024-07-04")
        # data = mtsa_util.parse_data()
        #
        # for mtsa_result in data:
        #     print(mtsa_result.lts, mtsa_result.duration_iso)
