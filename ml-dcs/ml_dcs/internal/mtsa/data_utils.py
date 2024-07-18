import json
import os
from logging import getLogger
from typing import Iterator

from ml_dcs.domain.mtsa import MTSAResult

logger = getLogger(__name__)


class MTSADataUtil:
    def __init__(self, input_dir_path: str):
        self.input_dir_path = input_dir_path
        self.ALLOWED_PREDICTION_TARGET = ["duration", "max_memory_usage"]

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

    # TODO
    def get_dataframe(self, target: str):
        if target not in self.ALLOWED_PREDICTION_TARGET:
            logger.error(f"Target {target} is not supported.")
            raise ValueError(f"Target {target} is not supported.")
