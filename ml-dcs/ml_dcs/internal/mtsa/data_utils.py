import json
import os
from logging import getLogger
from typing import Iterator, List

from ml_dcs.domain.dataset import ParsedDataset
from ml_dcs.domain.mtsa import MTSAResult
from ml_dcs.domain.mtsa_bench import MTSABenchResult

logger = getLogger(__name__)


class MTSADataUtil:
    def __init__(self, input_dir_path: str):
        self.input_dir_path = input_dir_path
        self.training_result_dir_path = os.path.join(input_dir_path, "training")
        self.validation_result_dir_path = os.path.join(input_dir_path, "validation")
        self.test_result_dir_path = os.path.join(input_dir_path, "testing")

    def _load_data(self, dir_path: str) -> Iterator[dict]:
        for obj in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, obj)):
                if obj.endswith(".json"):
                    with open(os.path.join(dir_path, obj), "r") as f:
                        data = json.load(f)
                        yield data

    def _get_parsed_data(self, path: str) -> List[MTSAResult]:
        data = []
        for dict_data in self._load_data(path):
            mtsa_result = MTSAResult(**dict_data)
            data.append(mtsa_result)
        return data

    def get_parsed_dataset(self) -> ParsedDataset:
        training_data = self._get_parsed_data(self.training_result_dir_path)
        validation_data = self._get_parsed_data(self.validation_result_dir_path)
        testing_data = self._get_parsed_data(self.test_result_dir_path)
        return ParsedDataset(
            training_data=training_data,
            validation_data=validation_data,
            testing_data=testing_data,
        )


class MTSABenchDataUtil:
    def __init__(self, bench_result_file_path: str):
        self.bench_result_file_path = bench_result_file_path

    def _load_data(self) -> dict:
        with open(self.bench_result_file_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _parse_data(self) -> MTSABenchResult:
        return MTSABenchResult(**self._load_data())
