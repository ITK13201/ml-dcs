import argparse
import json
import os
from logging import getLogger
from typing import List

from sklearn.model_selection import train_test_split

from ml_dcs.cmd.base import BaseCommand
from ml_dcs.domain.dataset import File
from ml_dcs.domain.mtsa import MTSAResult

logger = getLogger(__name__)
DEFAULT_RANDOM_STATE = 42


class PrepareDatasetCommand(BaseCommand):
    name = "prepare_dataset"
    help = "Prepare dataset"

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-i",
            "--input-dir",
            type=str,
            required=True,
            help="Input dataset directory",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            required=True,
            help="Output dataset directory",
        )
        parser.add_argument(
            "--calculation-time-threshold",
            type=float,
            required=False,
            help="Calculation time threshold (min)",
        )
        parser.add_argument(
            "--memory-usage-threshold",
            type=float,
            required=False,
            help="Memory usage threshold (GB)",
        )
        parser.add_argument(
            "--testing-scenario",
            type=str,
            required=False,
            help="Testing scenario",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # args
        self.input_dir = None
        self.output_dir = None
        self.calculation_time_threshold_ms = None
        self.memory_usage_threshold_kb = None
        self.testing_scenario = None
        # additional
        self.output_training_dataset_dir = None
        self.output_validation_dataset_dir = None
        self.output_testing_dataset_dir = None

    def execute(self, args: argparse.Namespace):
        logger.info("PrepareDatasetCommand started")

        # args
        self.input_dir: str = args.input_dir
        self.output_dir: str = args.output_dir
        self.calculation_time_threshold_ms: float | None = (
            args.calculation_time_threshold * 60 * 1000
            if args.calculation_time_threshold
            else None
        )
        self.memory_usage_threshold_kb: float | None = (
            args.memory_usage_threshold * 1000 * 1000
            if args.memory_usage_threshold
            else None
        )
        self.testing_scenario = args.testing_scenario

        # output dirs
        self.output_training_dataset_dir: str = os.path.join(
            self.output_dir, "training"
        )
        self.output_validation_dataset_dir: str = os.path.join(
            self.output_dir, "validation"
        )
        self.output_testing_dataset_dir: str = os.path.join(self.output_dir, "testing")
        # create dirs
        os.makedirs(self.output_training_dataset_dir, exist_ok=True)
        os.makedirs(self.output_validation_dataset_dir, exist_ok=True)
        os.makedirs(self.output_testing_dataset_dir, exist_ok=True)

        # ===

        logger.info("loading datasets...")
        dataset = self._load_datasets()

        logger.info("excluding by threshold from datasets...")
        dataset = self._exclude_by_threshold(dataset)

        logger.info("preparing dataset...")
        if not self.testing_scenario:
            training_dataset, tmp = train_test_split(
                dataset, test_size=0.3, random_state=DEFAULT_RANDOM_STATE, shuffle=True
            )
            validation_dataset, testing_dataset = train_test_split(
                tmp, test_size=0.5, random_state=DEFAULT_RANDOM_STATE, shuffle=True
            )
        else:
            not_testing_dataset, testing_dataset = self._split_dataset_by_scenario(dataset)
            training_dataset, validation_dataset = train_test_split(
                not_testing_dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, shuffle=True
            )

        logger.info("dumping datasets...")
        self._dump_dataset(self.output_training_dataset_dir, training_dataset)
        self._dump_dataset(self.output_validation_dataset_dir, validation_dataset)
        self._dump_dataset(self.output_testing_dataset_dir, testing_dataset)
        logger.info("done")

        logger.info("PrepareDatasetCommand finished")

    def _load_datasets(self) -> List[File]:
        dataset = []
        for obj in os.listdir(self.input_dir):
            if os.path.isfile(os.path.join(self.input_dir, obj)):
                if obj.endswith(".json"):
                    with open(os.path.join(self.input_dir, obj), "r") as f:
                        filename = os.path.basename(obj)
                        data = json.load(f)
                        dataset.append(
                            File(filename=filename, data=data, model=MTSAResult(**data))
                        )

        return dataset

    def _exclude_by_threshold(self, dataset: List[File]) -> List[File]:
        updated = []
        for obj in dataset:
            if self.calculation_time_threshold_ms is not None:
                if obj.model.duration_ms > self.calculation_time_threshold_ms:
                    continue
            if self.memory_usage_threshold_kb is not None:
                if obj.model.max_memory_usage_kb > self.memory_usage_threshold_kb:
                    continue
            updated.append(obj)
        return updated

    def _split_dataset_by_scenario(self, dataset: List[File]) -> (List[File], List[File]):
        if not self.testing_scenario:
            raise ValueError("No testing scenario specified")
        not_testing_dataset = []
        testing_dataset = []
        for obj in dataset:
            if obj.model.lts.startswith(self.testing_scenario):
                testing_dataset.append(obj)
            else:
                not_testing_dataset.append(obj)
        return not_testing_dataset, testing_dataset


    def _dump_dataset(self, dir: str, files: List[File]):
        for file in files:
            path = os.path.join(dir, file.filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(file.data, ensure_ascii=False, indent=2))
