from pydantic import BaseModel

from ml_dcs.domain.mtsa import MTSAResult


class File(BaseModel):
    filename: str
    data: dict
    model: MTSAResult
