import json
from logging import getLogger

from ml_dcs.domain.evaluation import EvaluationResult
from ml_dcs.domain.ml_gnn import GNNTestingResult
from ml_dcs.domain.ml_simple import MLSimpleTestingResultSet


logger = getLogger(__name__)

class AccuracyEvaluator:
    def __init__(self, simple_testing_result_path: str, gnn_testing_result_path: str):
        # args
        self.simple_testing_result_path = simple_testing_result_path
        self.gnn_testing_result_path = gnn_testing_result_path

        # data models
        self.simple_testing_result_set = self._load_simple_testing_result()
        self.gnn_testing_result = self._load_gnn_testing_result()

    def _load_simple_testing_result(self) -> MLSimpleTestingResultSet:
        with open(self.simple_testing_result_path, 'r') as f:
            data_dict = json.load(f)
        return MLSimpleTestingResultSet(**data_dict)

    def _load_gnn_testing_result(self) -> GNNTestingResult:
        with open(self.gnn_testing_result_path, 'r') as f:
            data_dict = json.load(f)
        return GNNTestingResult(**data_dict)

    def evaluate(self, threshold: float) -> (EvaluationResult, EvaluationResult):
        updated_actual_values = []
        updated_predicted_values = []
        for index, actual_value in enumerate(self.simple_testing_result_set.result_at_best_accuracy.actual_values):
            if actual_value < threshold:
                updated_predicted_value = self.simple_testing_result_set.result_at_best_accuracy.predicted_values[index]
                updated_actual_values.append(actual_value)
                updated_predicted_values.append(updated_predicted_value)
        updated_simple_result = EvaluationResult(
            actual_values=updated_actual_values,
            predicted_values=updated_predicted_values
        )

        updated_actual_values = []
        updated_predicted_values = []
        for index, actual_value in enumerate(self.gnn_testing_result.actual_values):
            if actual_value < threshold:
                updated_predicted_value = self.gnn_testing_result.predicted_values[index]
                updated_actual_values.append(actual_value)
                updated_predicted_values.append(updated_predicted_value)
        updated_gnn_result = EvaluationResult(
            actual_values=updated_actual_values,
            predicted_values=updated_predicted_values
        )

        logger.info("accuracy evaluate: %s", json.dumps(
            {
                "threshold": threshold,
                "simple_r2": updated_simple_result.r_squared,
                "gnn_r2": updated_gnn_result.r_squared,
            }
        ))

        return updated_simple_result, updated_gnn_result
