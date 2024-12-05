from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_dcs.domain.evaluation import EvaluationTarget
from ml_dcs.domain.ml_gnn import EdgeAttribute, ModelFeature, NodeFeature
from ml_dcs.domain.ml_simple import (
    MLCalculationTimePredictionInput2,
    MLMemoryUsagePredictionInput2,
    TestingDataSet,
    TrainingDataSet,
)
from ml_dcs.domain.mtsa import (
    MTSAResult,
    MTSAResultInitialModelsEnvironment,
    MTSAResultInitialModelsRequirement,
)


class MLSimplePreprocessor:
    def __init__(
        self,
        training_results: List[MTSAResult],
        testing_results: List[MTSAResult],
        target: EvaluationTarget,
    ):
        # args
        self.training_results = training_results
        self.testing_results = testing_results
        self.target = target

        # parameters
        self.scaler = StandardScaler()
        match target:
            case EvaluationTarget.CALCULATION_TIME:
                self.input_class = MLCalculationTimePredictionInput2
            case EvaluationTarget.MEMORY_USAGE:
                self.input_class = MLMemoryUsagePredictionInput2
            case _:
                raise RuntimeError(f"Unsupported evaluation target: {target}")

    def preprocess(self) -> Tuple[TrainingDataSet, TestingDataSet]:
        training_dataset = self._get_training_dataset()
        testing_dataset = self._get_testing_dataset()
        self.scaler.fit(training_dataset.x)
        training_dataset.x = self.scaler.transform(training_dataset.x)
        testing_dataset.x = self.scaler.transform(testing_dataset.x)
        return training_dataset, testing_dataset

    def _get_dataset(self, results: List[MTSAResult]) -> dict:
        lts_names = []
        data = []
        for result in results:
            lts_names.append(result.lts)
            input_model = self.input_class.init_by_mtsa_result(result)
            data.append(input_model)
        dataframe = pd.json_normalize(data)
        return dict(
            lts_names=lts_names,
            x=dataframe.iloc[:, :-1],
            y=dataframe.iloc[:, -1],
        )

    def _get_training_dataset(self) -> TrainingDataSet:
        return TrainingDataSet(**self._get_dataset(self.training_results))

    def _get_testing_dataset(self) -> TestingDataSet:
        return TestingDataSet(**self._get_dataset(self.testing_results))


class MinMaxScaler:
    def __init__(self, min, max, feature_range=(0, 1)):
        self.min = min
        self.max = max
        self.feature_range = feature_range

    def scale(self, value):
        value_range = self.max - self.min
        if value_range == 0:
            return value

        range_min, range_max = self.feature_range
        normalized_value = (value - self.min) / value_range
        return normalized_value * (range_max - range_min) + range_min


class LTSStructurePreprocessor:
    def preprocess(
        self, results: List[MTSAResult], standardize: bool = False
    ) -> List[List[ModelFeature]]:
        result_features = []
        for result in results:
            actions: Dict[str, int] = self._get_all_actions(result)
            environment_model_number_scaler = self._get_environment_model_number_scaler(
                result
            )
            requirement_model_number_scaler = self._get_requirement_model_number_scaler(
                result
            )
            model_features = []

            model_index = 1
            for environment in result.initial_models.environments:
                if standardize:
                    model_number = environment_model_number_scaler.scale(model_index)
                else:
                    model_number = model_index
                model_feature = self._get_model_feature(
                    model=environment,
                    is_environment=True,
                    model_number=model_number,
                    actions=actions,
                )
                model_features.append(model_feature)
                model_index += 1

            model_index = -1
            for requirement in result.initial_models.requirements:
                if standardize:
                    model_number = requirement_model_number_scaler.scale(model_index)
                else:
                    model_number = model_index
                model_feature = self._get_model_feature(
                    model=requirement,
                    is_environment=False,
                    model_number=model_number,
                    actions=actions,
                )
                model_features.append(model_feature)
                model_index -= 1
            result_features.append(model_features)
        return result_features

    def _get_environment_model_number_scaler(self, result: MTSAResult) -> MinMaxScaler:
        scaler = MinMaxScaler(
            min=1, max=result.initial_models.number_of_models_of_environments
        )
        return scaler

    def _get_requirement_model_number_scaler(self, result: MTSAResult) -> MinMaxScaler:
        scaler = MinMaxScaler(
            min=-result.initial_models.number_of_models_of_requirements,
            max=0,
            feature_range=(-1, 0),
        )
        return scaler

    def _get_all_actions(self, result: MTSAResult) -> Dict[str, int]:
        actions = set()
        for environment in result.initial_models.environments:
            for transition in environment.structure:
                action_name = transition[2]
                actions.add(action_name)
        for requirement in result.initial_models.requirements:
            for transition in requirement.structure:
                action_name = transition[2]
                actions.add(action_name)

        actions_dict = {action_name: index for index, action_name in enumerate(actions)}
        action_scaler = MinMaxScaler(0, len(actions_dict) - 1)
        actions_dict = {
            action_name: action_scaler.scale(action_number)
            for action_name, action_number in actions_dict.items()
        }
        return actions_dict

    def _get_model_feature(
        self,
        model: MTSAResultInitialModelsEnvironment | MTSAResultInitialModelsRequirement,
        is_environment: bool,
        model_number: int,
        actions: Dict[str, int],
        standardize: bool = False,
    ) -> ModelFeature:
        node_features = []
        state_number_scaler = MinMaxScaler(0, model.number_of_states - 1)
        for index in range(model.number_of_states):
            is_not_error_state = 1
            is_start_state = 0
            if index == 0:
                is_start_state = 1
            elif index == model.number_of_states - 1:
                is_not_error_state = -1
            if standardize:
                state_number = state_number_scaler.scale(index)
            else:
                state_number = index

            node_feature = NodeFeature(
                input_model_type=0 if is_environment else 1,
                input_model_number=model_number,
                state_number=state_number,
                is_not_error_state=is_not_error_state,
                is_start_state=is_start_state,
            )
            node_features.append(node_feature.get_array())

        edge_indexes = []
        edge_attributes = []
        for transition in model.structure:
            src_state_number = int(transition[0])
            dst_state_number = int(transition[1])
            action_name = transition[2]
            is_controllable = int(transition[3])
            if dst_state_number == -1:
                dst_state_number = model.number_of_states - 1
            forward_edge_attr = EdgeAttribute(
                action=actions[action_name],
                is_controllable=int(is_controllable),
                is_forward=1,
            )
            backward_edge_attr = EdgeAttribute(
                action=actions[action_name],
                is_controllable=int(is_controllable),
                is_forward=-1,
            )
            edge_attributes.append(forward_edge_attr.get_array())
            edge_indexes.append([src_state_number, dst_state_number])
            edge_attributes.append(backward_edge_attr.get_array())
            edge_indexes.append([dst_state_number, src_state_number])

        model_feature = ModelFeature(
            node_features=node_features,
            edge_indexes=edge_indexes,
            edge_attributes=edge_attributes,
        )
        return model_feature
