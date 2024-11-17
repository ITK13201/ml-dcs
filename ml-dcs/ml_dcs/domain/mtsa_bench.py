from datetime import datetime, timedelta
from typing import Dict, List

from pydantic import BaseModel, ByteSize, ConfigDict, Field, field_validator


class MTSABenchResultTask(BaseModel):
    name: str
    success: bool
    started_at: datetime
    finished_at: datetime
    max_memory_usage: ByteSize = Field(alias="max_memory_usage [KiB]")
    duration: timedelta

    @field_validator("max_memory_usage", mode="before", check_fields=True)
    @classmethod
    def validate_max_memory_usage(cls, v: int) -> ByteSize:
        kib = 2**10
        return ByteSize(v * kib)

    @property
    def max_memory_usage_kb(self) -> float:
        return self.max_memory_usage.to("KB")


class MTSABenchResult(BaseModel):
    started_at: datetime
    finished_at: datetime
    duration: timedelta

    tasks: List[MTSABenchResultTask]
    task_count: int
    task_success_count: int
    task_failure_count: int

    model_config = ConfigDict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tasks_dict = None

    @property
    def tasks_dict(self) -> Dict[str, MTSABenchResultTask]:
        if self._tasks_dict is None:
            data = {}
            for task in self.tasks:
                data[task.name] = task
            self._tasks_dict = data
            return self._tasks_dict
        else:
            return self._tasks_dict
