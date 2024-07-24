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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_duration = None

    @field_validator("compose_duration", mode="before", check_fields=True)
    @classmethod
    def validate_compose_duration(cls, v: int) -> timedelta:
        return timedelta(milliseconds=float(v))

    @field_validator("solving_duration", mode="before", check_fields=True)
    @classmethod
    def validate_solving_duration(cls, v: int) -> timedelta:
        return timedelta(milliseconds=float(v))

    @property
    def total_duration(self) -> timedelta:
        if self._total_duration is None:
            self._total_duration = self.compose_duration + self.solving_duration
        return self._total_duration


# ===
# classes by MTSAResult
# ===
class MTSAResultCompileStep(BaseModel):
    environments: List[MTSAResultCompileStepEnvironment] = Field(default_factory=list)
    requirements: List[MTSAResultCompileStepRequirement] = Field(default_factory=list)
    final_models: List[MTSAResultCompileStepFinalModel] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_compose_duration_of_environments = None
        self._total_minimize_duration_of_requirements = None
        self._total_duration = None
        self._total_number_of_states = None
        self._total_number_of_transitions = None
        self._total_number_of_controllable_actions = None
        self._total_number_of_uncontrollable_actions = None
        self._ratio_of_controllable_actions = None
        self._number_of_models = None

    @property
    def total_compose_duration_of_environments(self) -> timedelta:
        if self._total_compose_duration_of_environments is None:
            result = timedelta(microseconds=0)
            for environment in self.environments:
                result += environment.compose_duration
            self._total_compose_duration_of_environments = result
        return self._total_compose_duration_of_environments

    @property
    def total_minimize_duration_of_requirements(self) -> timedelta:
        if self._total_minimize_duration_of_requirements is None:
            result = timedelta(microseconds=0)
            for requirement in self.requirements:
                result += requirement.minimize_duration
            self._total_minimize_duration_of_requirements = result
        return self._total_minimize_duration_of_requirements

    @property
    def total_duration(self) -> timedelta:
        if self._total_duration is None:
            self._total_duration = (
                self.total_compose_duration_of_environments
                + self.total_minimize_duration_of_requirements
            )
        return self._total_duration

    @property
    def total_number_of_states(self) -> int:
        if self._total_number_of_states is None:
            result = 0
            for model in self.final_models:
                result += model.number_of_states
            self._total_number_of_states = result
        return self._total_number_of_states

    @property
    def total_number_of_transitions(self) -> int:
        if self._total_number_of_transitions is None:
            result = 0
            for model in self.final_models:
                result += model.number_of_transitions
            self._total_number_of_transitions = result
        return self._total_number_of_transitions

    @property
    def total_number_of_controllable_actions(self) -> int:
        if self._total_number_of_controllable_actions is None:
            result = 0
            for model in self.final_models:
                result += model.number_of_controllable_actions
            self._total_number_of_controllable_actions = result
        return self._total_number_of_controllable_actions

    @property
    def total_number_of_uncontrollable_actions(self) -> int:
        if self._total_number_of_uncontrollable_actions is None:
            result = 0
            for model in self.final_models:
                result += model.number_of_uncontrollable_actions
            self._total_number_of_uncontrollable_actions = result
        return self._total_number_of_uncontrollable_actions

    @property
    def ratio_of_controllable_actions(self) -> float:
        if self._ratio_of_controllable_actions is None:
            self._ratio_of_controllable_actions = (
                self.total_number_of_controllable_actions
                / self.total_number_of_transitions
            )
        return self._ratio_of_controllable_actions

    @property
    def number_of_models(self) -> int:
        if self._number_of_models is None:
            self._number_of_models = len(self.final_models)
        return self._number_of_models


class MTSAResultComposeStep(BaseModel):
    creatingGameSpace: MTSAResultComposeStepCreatingGameSpace = Field(
        default_factory=None
    )
    solvingProblem: MTSAResultComposeStepSolvingProblem = Field(default_factory=None)

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_duration = None

    @property
    def total_duration(self) -> timedelta:
        if self._total_duration is None:
            self._total_duration = (
                self.creatingGameSpace.compose_duration
                + self.solvingProblem.total_duration
            )
        return self._total_duration


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._significant_duration = None

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

    @property
    def significant_duration(self):
        if self._significant_duration is None:
            self._significant_duration = (
                self.compile_step.total_duration + self.compose_step.total_duration
            )
        return self._significant_duration
