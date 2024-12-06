from typing import List

from pydantic import BaseModel

from ml_dcs.domain.mtsa import MTSAResult


class File(BaseModel):
    filename: str
    data: dict
    model: MTSAResult


class ParsedDataset(BaseModel):
    training_data: List[MTSAResult]
    validation_data: List[MTSAResult]
    testing_data: List[MTSAResult]
