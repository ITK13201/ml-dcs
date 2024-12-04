import json
from logging import getLogger

from ml_dcs.domain.evaluation import EvaluationResult
from ml_dcs.domain.ml_gnn import GNNTestingResult

logger = getLogger(__name__)

SCENARIOS = [
    "ArtGallery",
    "AT",
    "BW",
    "CM"
]

class ScenarioEvaluator:
    def __init__(self, gnn_testing_result_path: str):
        self.gnn_testing_result_path = gnn_testing_result_path
        self.gnn_testing_result = self._load_gnn_testing_result()

    def _load_gnn_testing_result(self) -> GNNTestingResult:
        with open(self.gnn_testing_result_path, 'r') as f:
            data_dict = json.load(f)
        return GNNTestingResult(**data_dict)

    def evaluate(self, scenario: str) -> EvaluationResult:
        if scenario not in SCENARIOS:
            raise ValueError(f"Scenario {scenario} is not supported")

        updated_actual_values = []
        updated_predicted_values = []
        for task_result in self.gnn_testing_result.task_results:
            if task_result.lts_name.startswith(scenario):
                updated_actual_values.append(task_result.actual)
                updated_predicted_values.append(task_result.predicted)
        updated_gnn_result = EvaluationResult(
            actual_values=updated_actual_values,
            predicted_values=updated_predicted_values
        )

        logger.info("accuracy evaluate: %s", json.dumps(
            {
                "scenario": scenario,
                "r2": updated_gnn_result.r_squared,
                "mae": updated_gnn_result.mae,
            }
        ))

        return updated_gnn_result
