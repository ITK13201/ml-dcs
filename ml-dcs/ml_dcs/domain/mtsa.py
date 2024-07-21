from datetime import datetime, timedelta
from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ByteSize,
    field_validator,
    TypeAdapter,
)
from pydantic.alias_generators import to_camel


# ===
# classes by MTSAResultCompileStep
# ===
class MTSAResultCompileStepEnvironment(BaseModel):
    name: str
    number_of_max_states: int
    number_of_states: int
    number_of_transitions: int
    compose_duration: timedelta = Field(alias="composeDuration [ms]")
    source_models: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)

    @field_validator("compose_duration", mode="before", check_fields=True)
    @classmethod
    def validate_compose_duration(cls, v: int) -> timedelta:
        return timedelta(milliseconds=float(v))


class MTSAResultCompileStepRequirement(BaseModel):
    name: str
    minimize_duration: timedelta = Field(alias="minimizeDuration [ms]")

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)

    @field_validator("minimize_duration", mode="before", check_fields=True)
    @classmethod
    def validate_minimize_duration(cls, v: int) -> timedelta:
        return timedelta(milliseconds=float(v))


class MTSAResultCompileStepFinalModel(BaseModel):
    name: str
    number_of_states: int
    number_of_transitions: int
    number_of_controllable_actions: int
    number_of_uncontrollable_actions: int

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)


# ===
# classes by MTSAResultComposeStep
# ===
class MTSAResultComposeStepCreatingGameSpace(BaseModel):
    number_of_max_states: int
    number_of_states: int
    number_of_transitions: int
    number_of_controllable_actions: int
    number_of_uncontrollable_actions: int
    compose_duration: timedelta = Field(alias="composeDuration [ms]")
    source_models: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)

    @field_validator("compose_duration", mode="before", check_fields=True)
    @classmethod
    def validate_compose_duration(cls, v: int) -> timedelta:
        return timedelta(milliseconds=float(v))


class MTSAResultComposeStepSolvingProblem(BaseModel):
    number_of_max_states: int
    number_of_states: int
    number_of_transitions: int
    number_of_controllable_actions: int
    number_of_uncontrollable_actions: int
    compose_duration: timedelta = Field(alias="composeDuration [ms]")
    solving_duration: timedelta = Field(alias="solvingDuration [ms]")
    source_models: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)

    @field_validator("compose_duration", mode="before", check_fields=True)
    @classmethod
    def validate_compose_duration(cls, v: int) -> timedelta:
        return timedelta(milliseconds=float(v))

    @field_validator("solving_duration", mode="before", check_fields=True)
    @classmethod
    def validate_solving_duration(cls, v: int) -> timedelta:
        return timedelta(milliseconds=float(v))


# ===
# classes by MTSAResult
# ===
class MTSAResultCompileStep(BaseModel):
    environments: List[MTSAResultCompileStepEnvironment] = Field(default_factory=list)
    requirements: List[MTSAResultCompileStepRequirement] = Field(default_factory=list)
    final_models: List[MTSAResultCompileStepFinalModel] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)


class MTSAResultComposeStep(BaseModel):
    creatingGameSpace: MTSAResultComposeStepCreatingGameSpace = Field(
        default_factory=None
    )
    solvingProblem: MTSAResultComposeStepSolvingProblem = Field(default_factory=None)

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)


# ===
# main Class
# ===
class MTSAResult(BaseModel):
    mode: str
    command: str
    lts_file_path: str
    target: str
    lts: str

    compile_step: MTSAResultCompileStep = Field(default_factory=None)
    compose_step: MTSAResultComposeStep = Field(default_factory=None)

    started_at: datetime
    finished_at: datetime
    duration: timedelta = Field(alias="duration [ms]")
    max_memory_usage: ByteSize = Field(alias="maxMemoryUsage [KiB]")

    # configuration of this model
    model_config = ConfigDict(frozen=True, alias_generator=to_camel)

    @field_validator("duration", mode="before", check_fields=True)
    @classmethod
    def validate_duration(cls, v: int) -> timedelta:
        return timedelta(milliseconds=float(v))

    @field_validator("max_memory_usage", mode="before", check_fields=True)
    @classmethod
    def validate_max_memory_usage(cls, v: int) -> ByteSize:
        kib = 2**10
        return ByteSize(v * kib)

    @property
    def duration_iso(self) -> str:
        adapter = TypeAdapter(timedelta)
        return adapter.dump_python(self.duration, mode="json")
