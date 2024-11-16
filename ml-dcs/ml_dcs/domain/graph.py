from typing import List, Tuple

from pydantic import BaseModel

from ml_dcs.domain.ml_gnn import GNNTestingResult
from ml_dcs.domain.ml_simple import MLSimpleTestingResultSet


class GraphData(BaseModel):
    x: List[float]
    y: List[float]


class PredictionAccuracyGraphData(GraphData):
    @classmethod
    def from_ml_simple_class(cls, data: MLSimpleTestingResultSet) -> "GraphData":
        return cls(
            x=data.result_at_best_accuracy.actual_values,
            y=data.result_at_best_accuracy.predicted_values,
        )

    @classmethod
    def from_ml_gnn_class(cls, data: GNNTestingResult) -> "GraphData":
        return cls(
            x=data.actual_values,
            y=data.predicted_values,
        )


class Graph(BaseModel):
    data: PredictionAccuracyGraphData
    x_label: str
    y_label: str
    x_lim: Tuple[float, float] | None = None
    y_lim: Tuple[float, float] | None = None
    order_of_mag: int | None = None
    output_path: str | None = None
