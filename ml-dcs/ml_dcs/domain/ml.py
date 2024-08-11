from abc import ABC, abstractmethod

from pydantic import BaseModel

from ml_dcs.domain.mtsa import MTSAResult


class BaseMLInput(ABC, BaseModel):
    @classmethod
    @abstractmethod
    def init_by_mtsa_result(cls, mtsa_result: MTSAResult) -> "BaseMLInput":
        pass


# ===
# CALCULATION TIME
# ===
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


class MLCalculationTimePredictionInput2(BaseMLInput):
    number_of_models_of_environments: int
    max_number_of_states_of_environments: int
    sum_of_number_of_transitions_of_environments: int
    sum_of_number_of_controllable_actions_of_environments: int
    sum_of_number_of_uncontrollable_actions_of_environments: int
    number_of_models_of_requirements: int
    sum_of_number_of_states_of_requirements: int
    sum_of_number_of_transitions_of_requirements: int
    sum_of_number_of_controllable_actions_of_requirements: int
    sum_of_number_of_uncontrollable_actions_of_requirements: int

    # milliseconds (ms)
    calculation_time: float

    @classmethod
    def init_by_mtsa_result(
        cls, mtsa_result: MTSAResult
    ) -> "MLCalculationTimePredictionInput2":
        variables = {
            "number_of_models_of_environments": mtsa_result.initial_models.number_of_models_of_environments,
            "max_number_of_states_of_environments": mtsa_result.initial_models.max_number_of_states_of_environments,
            "sum_of_number_of_transitions_of_environments": mtsa_result.initial_models.sum_of_number_of_transitions_of_environments,
            "sum_of_number_of_controllable_actions_of_environments": mtsa_result.initial_models.sum_of_number_of_controllable_actions_of_environments,
            "sum_of_number_of_uncontrollable_actions_of_environments": mtsa_result.initial_models.sum_of_number_of_uncontrollable_actions_of_environments,
            "number_of_models_of_requirements": mtsa_result.initial_models.number_of_models_of_requirements,
            "sum_of_number_of_states_of_requirements": mtsa_result.initial_models.sum_of_number_of_states_of_requirements,
            "sum_of_number_of_transitions_of_requirements": mtsa_result.initial_models.sum_of_number_of_transitions_of_requirements,
            "sum_of_number_of_controllable_actions_of_requirements": mtsa_result.initial_models.sum_of_number_of_controllable_actions_of_requirements,
            "sum_of_number_of_uncontrollable_actions_of_requirements": mtsa_result.initial_models.sum_of_number_of_uncontrollable_actions_of_requirements,
            "calculation_time": mtsa_result.duration_ms,
        }
        return cls(**variables)


# ===
# MEMORY USAGE
# ===
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


class MLMemoryUsagePredictionInput2(BaseMLInput):
    number_of_models_of_environments: int
    max_number_of_states_of_environments: int
    sum_of_number_of_transitions_of_environments: int
    sum_of_number_of_controllable_actions_of_environments: int
    sum_of_number_of_uncontrollable_actions_of_environments: int
    number_of_models_of_requirements: int
    sum_of_number_of_states_of_requirements: int
    sum_of_number_of_transitions_of_requirements: int
    sum_of_number_of_controllable_actions_of_requirements: int
    sum_of_number_of_uncontrollable_actions_of_requirements: int

    # KiB
    memory_usage: int

    @classmethod
    def init_by_mtsa_result(
        cls, mtsa_result: MTSAResult
    ) -> "MLMemoryUsagePredictionInput2":
        variables = {
            "number_of_models_of_environments": mtsa_result.initial_models.number_of_models_of_environments,
            "max_number_of_states_of_environments": mtsa_result.initial_models.max_number_of_states_of_environments,
            "sum_of_number_of_transitions_of_environments": mtsa_result.initial_models.sum_of_number_of_transitions_of_environments,
            "sum_of_number_of_controllable_actions_of_environments": mtsa_result.initial_models.sum_of_number_of_controllable_actions_of_environments,
            "sum_of_number_of_uncontrollable_actions_of_environments": mtsa_result.initial_models.sum_of_number_of_uncontrollable_actions_of_environments,
            "number_of_models_of_requirements": mtsa_result.initial_models.number_of_models_of_requirements,
            "sum_of_number_of_states_of_requirements": mtsa_result.initial_models.sum_of_number_of_states_of_requirements,
            "sum_of_number_of_transitions_of_requirements": mtsa_result.initial_models.sum_of_number_of_transitions_of_requirements,
            "sum_of_number_of_controllable_actions_of_requirements": mtsa_result.initial_models.sum_of_number_of_controllable_actions_of_requirements,
            "sum_of_number_of_uncontrollable_actions_of_requirements": mtsa_result.initial_models.sum_of_number_of_uncontrollable_actions_of_requirements,
            "memory_usage": mtsa_result.max_memory_usage_kib,
        }
        return cls(**variables)


class MLResult(BaseModel):
    r2_score: float
    mae: float
