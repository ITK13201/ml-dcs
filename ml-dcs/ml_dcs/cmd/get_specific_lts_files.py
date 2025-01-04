import argparse
import glob
import json
import os
import shutil
from logging import getLogger

from ml_dcs.cmd.base import BaseCommand
from ml_dcs.domain.mtsa import MTSAResult

logger = getLogger(__name__)


class GetSpecificLTSFilesCommand(BaseCommand):
    name = "get_specific_lts_files"
    help = "Get specific LTS files for RERUN"

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-l",
            "--lts-input-dir",
            type=str,
            required=True,
            help="LTS file input directory",
        )
        parser.add_argument(
            "-r",
            "--result-input-dir",
            type=str,
            required=True,
            help="MTSA result file input directory",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            required=True,
            help="Output directory",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # args
        self.lts_input_dir = None
        self.result_input_dir = None
        self.output_dir = None

    def execute(self, args: argparse.Namespace):
        logger.info("GetSpecificLTSFiles started")

        # args
        self.lts_input_dir: str = args.lts_input_dir
        self.result_input_dir: str = args.result_input_dir
        self.output_dir: str = args.output_dir

        # search in MTSA result file dir
        for result_file_path in glob.glob(
            f"{self.result_input_dir}/**/*.json", recursive=True
        ):
            with open(result_file_path, "r") as f:
                data = json.load(f)
            data_model = MTSAResult(**data)
            lts_file_path = os.path.join(self.lts_input_dir, data_model.lts + ".lts")
            if os.path.exists(lts_file_path):
                logger.info(f"Found: {lts_file_path}")
                shutil.copy(lts_file_path, self.output_dir)
                logger.info(f"Copied: {lts_file_path}")
            else:
                logger.info(f"Not found: {lts_file_path}")
                raise RuntimeError("LTS file not found")

        logger.info("GetSpecificLTSFiles finished")
