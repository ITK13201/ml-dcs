import json
from logging import getLogger

from ml_dcs.domain.evaluation import EvaluationResult
from ml_dcs.domain.ml_gnn import GNNTestingResult
from ml_dcs.domain.ml_simple import MLSimpleTestingResultSet

logger = getLogger(__name__)

SCENARIOS = ["ArtGallery", "AT", "BW", "CM"]


class SimpleScenarioEvaluator:
    def __init__(self, simple_testing_result_path: str):
        self._simple_testing_result_path = simple_testing_result_path
        self.simple_testing_result_set = self._load_simple_testing_result()

    def _load_simple_testing_result(self) -> MLSimpleTestingResultSet:
        with open(self._simple_testing_result_path, "r") as f:
            data_dict = json.load(f)
        return MLSimpleTestingResultSet(**data_dict)

    def evaluate(self, scenario: str, threshold: float = None) -> EvaluationResult:
        if scenario not in SCENARIOS:
            raise ValueError(f"Scenario '{scenario}' is not supported.")

        updated_actual_values = []
        updated_predicted_values = []
        best_result = self.simple_testing_result_set.result_at_best_accuracy
        for index, lts_name in enumerate(best_result.lts_names):
            # exclude
            if not lts_name.startswith(scenario):
                continue
            actual_value = best_result.actual_values[index]
            if threshold is not None:
                if actual_value > threshold:
                    continue
            # include
            actual_value = best_result.actual_values[index]
            predicted_value = best_result.predicted_values[index]
            updated_actual_values.append(actual_value)
            updated_predicted_values.append(predicted_value)
        updated_simple_result = EvaluationResult(
            actual_values=updated_actual_values,
            predicted_values=updated_predicted_values,
        )

        logger.info(
            "accuracy evaluate: %s",
            json.dumps(
                {
                    "scenario": scenario,
                    "r2": updated_simple_result.r_squared,
                    "mae": updated_simple_result.mae,
                }
            ),
        )
        return updated_simple_result


class GNNScenarioEvaluator:
    def __init__(self, gnn_testing_result_path: str):
        self.gnn_testing_result_path = gnn_testing_result_path
        self.gnn_testing_result = self._load_gnn_testing_result()

    def _load_gnn_testing_result(self) -> GNNTestingResult:
        with open(self.gnn_testing_result_path, "r") as f:
            data_dict = json.load(f)
        return GNNTestingResult(**data_dict)

    def evaluate(self, scenario: str, threshold: float = None) -> EvaluationResult:
        if scenario not in SCENARIOS:
            raise ValueError(f"Scenario '{scenario}' is not supported")

        updated_actual_values = []
        updated_predicted_values = []
        for task_result in self.gnn_testing_result.task_results:
            # exclude
            if not task_result.lts_name.startswith(scenario):
                continue
            if threshold is not None:
                if task_result.actual > threshold:
                    continue
            # include
            updated_actual_values.append(task_result.actual)
            updated_predicted_values.append(task_result.predicted)
        updated_gnn_result = EvaluationResult(
            actual_values=updated_actual_values,
            predicted_values=updated_predicted_values,
        )

        logger.info(
            "accuracy evaluate: %s",
            json.dumps(
                {
                    "scenario": scenario,
                    "r2": updated_gnn_result.r_squared,
                    "mae": updated_gnn_result.mae,
                }
            ),
        )

        return updated_gnn_result
