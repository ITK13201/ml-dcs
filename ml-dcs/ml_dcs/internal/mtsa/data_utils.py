import json
import os
from logging import getLogger
from typing import Iterator, Type

import pandas as pd

from ml_dcs.domain.ml import BaseMLInput
from ml_dcs.domain.mtsa import MTSAResult
from ml_dcs.domain.mtsa_bench import MTSABenchResult

logger = getLogger(__name__)


class MTSADataUtil:
    def __init__(self, input_dir_path: str):
        self.input_dir_path = input_dir_path

    def _load_data(self) -> Iterator[dict]:
        for obj in os.listdir(self.input_dir_path):
            if os.path.isfile(os.path.join(self.input_dir_path, obj)):
                if obj.endswith(".json"):
                    with open(os.path.join(self.input_dir_path, obj), "r") as f:
                        data = json.load(f)
                        yield data

    def _parse_data(self) -> Iterator[MTSAResult]:
        dict_dataset = self._load_data()
        for dict_data in dict_dataset:
            mtsa_result = MTSAResult(**dict_data)
            yield mtsa_result

    def get_dataframe(self, ml_input_class: Type[BaseMLInput]) -> pd.DataFrame:
        data = []
        for parsed in self._parse_data():
            input_model = ml_input_class.init_by_mtsa_result(parsed)
            data.append(input_model.model_dump())
        return pd.json_normalize(data)


class MTSABenchDataUtil:
    def __init__(self, bench_result_file_path: str):
        self.bench_result_file_path = bench_result_file_path

    def _load_data(self) -> dict:
        with open(self.bench_result_file_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _parse_data(self) -> MTSABenchResult:
        return MTSABenchResult(**self._load_data())
