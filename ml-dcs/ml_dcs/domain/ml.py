from abc import ABC, abstractmethod

from pydantic import BaseModel

from ml_dcs.domain.mtsa import MTSAResult


class BaseMLInput(ABC, BaseModel):
    @classmethod
    @abstractmethod
    def init_by_mtsa_result(cls, mtsa_result: MTSAResult) -> "BaseMLInput":
        pass


class MLCalculationTimePredictionInput1(BaseMLInput):
    total_number_of_states: int
    total_number_of_transitions: int
    total_number_of_controllable_actions: int
    total_number_of_uncontrollable_actions: int
    # ratio_of_controllable_actions: float
    number_of_models: int

    # milliseconds (ms)
    calculation_time: float

    @classmethod
    def init_by_mtsa_result(
        cls, mtsa_result: MTSAResult
    ) -> "MLCalculationTimePredictionInput1":
        variables = {
            "total_number_of_states": mtsa_result.compile_step.total_number_of_states,
            "total_number_of_transitions": mtsa_result.compile_step.total_number_of_transitions,
            "total_number_of_controllable_actions": mtsa_result.compile_step.total_number_of_controllable_actions,
            "total_number_of_uncontrollable_actions": mtsa_result.compile_step.total_number_of_uncontrollable_actions,
            "ratio_of_controllable_actions": mtsa_result.compile_step.ratio_of_controllable_actions,
            "number_of_models": mtsa_result.compile_step.number_of_models,
            "calculation_time": mtsa_result.duration_ms,
        }
        return cls(**variables)


class MLMemoryUsagePredictionInput1(BaseMLInput):
    total_number_of_states: int
    total_number_of_transitions: int
    total_number_of_controllable_actions: int
    total_number_of_uncontrollable_actions: int
    # ratio_of_controllable_actions: float
    number_of_models: int

    # KiB
    memory_usage: int

    @classmethod
    def init_by_mtsa_result(
        cls, mtsa_result: MTSAResult
    ) -> "MLMemoryUsagePredictionInput1":
        variables = {
            "total_number_of_states": mtsa_result.compile_step.total_number_of_states,
            "total_number_of_transitions": mtsa_result.compile_step.total_number_of_transitions,
            "total_number_of_controllable_actions": mtsa_result.compile_step.total_number_of_controllable_actions,
            "total_number_of_uncontrollable_actions": mtsa_result.compile_step.total_number_of_uncontrollable_actions,
            "ratio_of_controllable_actions": mtsa_result.compile_step.ratio_of_controllable_actions,
            "number_of_models": mtsa_result.compile_step.number_of_models,
            "memory_usage": mtsa_result.max_memory_usage_kib,
        }
        return cls(**variables)


class MLResult(BaseModel):
    r2_score: float
    mae: float
