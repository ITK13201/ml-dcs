from datetime import datetime, timedelta
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class MTSABenchResultTask(BaseModel):
    name: str
    success: bool
    started_at: datetime
    finished_at: datetime
    max_memory_usage: float = Field(alias="max_memory_usage [KiB]")
    duration: timedelta


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
