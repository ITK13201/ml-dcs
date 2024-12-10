from typing import List, Tuple

from pydantic import BaseModel

from ml_dcs.domain.evaluation import EvaluationResult
from ml_dcs.domain.ml_gnn import GNNTestingResult, GNNTrainingResult
from ml_dcs.domain.ml_simple import MLSimpleTestingResultSet


class GraphData(BaseModel):
    x: List[float]
    y: List[float]


class GraphData2(BaseModel):
    x: List[float]
    y1: List[float]
    y2: List[float]


class PredictionAccuracyGraphData(GraphData):
    @classmethod
    def from_ml_simple_class(cls, data: MLSimpleTestingResultSet) -> "GraphData":
        return cls(
            x=data.result_at_best_accuracy.actual_values,
            y=data.result_at_best_accuracy.predicted_values,
        )

    @classmethod
    def from_ml_simple_class_div1000(
        cls, data: MLSimpleTestingResultSet
    ) -> "GraphData":
        return cls(
            x=data.result_at_best_accuracy.actual_values_div1000,
            y=data.result_at_best_accuracy.predicted_values_div1000,
        )

    @classmethod
    def from_ml_gnn_class(cls, data: GNNTestingResult) -> "GraphData":
        return cls(
            x=data.actual_values,
            y=data.predicted_values,
        )

    @classmethod
    def from_ml_gnn_class_div1000(cls, data: GNNTestingResult) -> "GraphData":
        return cls(
            x=data.actual_values_div1000,
            y=data.predicted_values_div1000,
        )

    @classmethod
    def from_evaluation_result_class(cls, data: EvaluationResult) -> "GraphData":
        return cls(
            x=data.actual_values,
            y=data.predicted_values,
        )

    @classmethod
    def from_evaluation_result_class_div1000(
        cls, data: EvaluationResult
    ) -> "GraphData":
        return cls(
            x=data.actual_values_div1000,
            y=data.predicted_values_div1000,
        )


class LearningCurveGraphData(GraphData2):
    @classmethod
    def from_ml_gnn_training_result(cls, data: GNNTrainingResult) -> "GraphData2":
        x = []
        y1 = []
        y2 = []
        for index, epoch_result in enumerate(data.epoch_results):
            epoch = index + 1
            train_loss = epoch_result.training_loss_avg
            val_loss = epoch_result.validation_loss_avg
            x.append(epoch)
            y1.append(train_loss)
            y2.append(val_loss)

        return cls(x=x, y1=y1, y2=y2)


class Graph(BaseModel):
    data: PredictionAccuracyGraphData
    x_label: str
    y_label: str
    x_lim: Tuple[float, float] | None = None
    y_lim: Tuple[float, float] | None = None
    order_of_mag: int | None = None
    output_path: str | None = None


class Graph2(Graph):
    data: LearningCurveGraphData
    y1_legend: str
    y2_legend: str
