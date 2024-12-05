import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool

from ml_dcs.config.config import DEVICE
from ml_dcs.domain.ml_gnn import ModelFeature, TestingData, TrainingData, ValidationData
from ml_dcs.domain.mtsa import MTSAResult
from ml_dcs.internal.preprocessor.preprocessor import LTSStructurePreprocessor

logger = logging.getLogger(__name__)

LTS_EMBEDDING_SIZE = 64
DEFAULT_RANDOM_STATE = 42


class LTSGNN(torch.nn.Module):
    INPUT_CHANNELS = 5
    HIDDEN_CHANNELS = 128
    OUTPUT_CHANNELS = LTS_EMBEDDING_SIZE
    EDGE_DIMENSION = 3

    def __init__(self):
        super(LTSGNN, self).__init__()
        self.conv1 = GATConv(
            self.INPUT_CHANNELS, self.HIDDEN_CHANNELS, edge_dim=self.EDGE_DIMENSION
        ).to(DEVICE)
        self.conv2 = GATConv(
            self.HIDDEN_CHANNELS, self.OUTPUT_CHANNELS, edge_dim=self.EDGE_DIMENSION
        ).to(DEVICE)

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # 1st graph convolution
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = F.relu(x)

        # 2nd graph convolution
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # pooling
        x = global_mean_pool(x, batch)
        return x


class LTSRegressionModel(torch.nn.Module):
    INPUT_CHANNELS = LTS_EMBEDDING_SIZE
    HIDDEN_CHANNELS = 32
    OUTPUT_CHANNELS = 1

    def __init__(self):
        super(LTSRegressionModel, self).__init__()
        self.fc1 = torch.nn.Linear(self.INPUT_CHANNELS, self.HIDDEN_CHANNELS).to(DEVICE)
        self.fc2 = torch.nn.Linear(self.HIDDEN_CHANNELS, self.OUTPUT_CHANNELS).to(
            DEVICE
        )

    def forward(self, x):
        # 1st regression
        x = self.fc1(x)
        x = F.relu(x)

        # 2nd regression
        x = self.fc2(x)
        return x


class GNNDataUtil:
    ALLOWED_TARGET_NAMES: List[str] = ["calculation_time", "memory_usage"]

    def __init__(
        self, mtsa_results: List[MTSAResult], target_name: str, threshold: float = None
    ):
        # args
        self.mtsa_results = mtsa_results
        if target_name in self.ALLOWED_TARGET_NAMES:
            self.target_name = target_name
        else:
            raise ValueError(f"Target name {target_name} not allowed")
        self.threshold = threshold
        # additional
        if self.threshold is not None:
            self.mtsa_results = self._exclude_by_threshold()
        (
            self.training_mtsa_results,
            self.validation_mtsa_results,
            self.testing_mtsa_results,
        ) = self._split_mtsa_results(self.mtsa_results)
        self.preprocessor = LTSStructurePreprocessor()

    def _exclude_by_threshold(self) -> List[MTSAResult]:
        updated = []
        for result in self.mtsa_results:
            match self.target_name:
                case "calculation_time":
                    if result.duration_ms > self.threshold:
                        continue
                case "memory_usage":
                    if result.max_memory_usage_kb > self.threshold:
                        continue
                case _:
                    raise ValueError(f"Target name {self.target_name} not allowed")
            updated.append(result)
        return updated

    def _split_mtsa_results(
        self, mtsa_results: List[MTSAResult]
    ) -> Tuple[List[MTSAResult], List[MTSAResult], List[MTSAResult]]:
        training_results, tmp = train_test_split(
            mtsa_results, test_size=0.3, random_state=DEFAULT_RANDOM_STATE
        )
        validation_results, testing_results = train_test_split(
            tmp, test_size=0.5, random_state=DEFAULT_RANDOM_STATE
        )
        return training_results, validation_results, testing_results

    def get_lts_graph(
        self,
        model_feature: ModelFeature,
    ) -> Data:
        node_features_tensor = torch.tensor(
            model_feature.node_features, dtype=torch.float
        ).to(DEVICE)
        edge_indexes_tensor = (
            torch.tensor(model_feature.edge_indexes, dtype=torch.long)
            .t()
            .contiguous()
            .to(DEVICE)
        )
        edge_attributes_tensor = torch.tensor(
            model_feature.edge_attributes, dtype=torch.float
        ).to(DEVICE)

        lts_embedding = Data(
            x=node_features_tensor,
            edge_index=edge_indexes_tensor,
            edge_weight=edge_attributes_tensor,
        ).to(DEVICE)

        return lts_embedding

    def get_lts_set_graph(self, result_feature: List[ModelFeature]) -> Batch:
        lts_embeddings = []
        for model_feature in result_feature:
            lts_embeddings.append(self.get_lts_graph(model_feature))

        batch = Batch.from_data_list(lts_embeddings)
        return batch

    def get_training_dataset(self) -> List[TrainingData]:
        dataset = []
        mtsa_results = self.training_mtsa_results
        result_features = self.preprocessor.preprocess(mtsa_results)
        for index, result_feature in enumerate(result_features):
            mtsa_result = mtsa_results[index]
            lts_set_graph = self.get_lts_set_graph(result_feature)
            match self.target_name:
                case "calculation_time":
                    target = mtsa_result.duration_ms
                case "memory_usage":
                    target = mtsa_result.max_memory_usage_kb
                case _:
                    raise ValueError(f"Unknown target name: {self.target_name}")
            target_tensor = torch.tensor(target).to(DEVICE)

            data = TrainingData(
                lts_name=mtsa_result.lts,
                lts_set_graph=lts_set_graph,
                target=target_tensor,
            )
            dataset.append(data)
        return dataset

    def get_validation_dataset(self) -> List[ValidationData]:
        dataset = []
        mtsa_results = self.validation_mtsa_results
        result_features = self.preprocessor.preprocess(mtsa_results)
        for index, result_feature in enumerate(result_features):
            mtsa_result = mtsa_results[index]
            lts_set_graph = self.get_lts_set_graph(result_feature)
            match self.target_name:
                case "calculation_time":
                    target = mtsa_result.duration_ms
                case "memory_usage":
                    target = mtsa_result.max_memory_usage_kb
                case _:
                    raise ValueError(f"Unknown target name: {self.target_name}")
            target_tensor = torch.tensor(target).to(DEVICE)

            data = ValidationData(
                lts_name=mtsa_result.lts,
                lts_set_graph=lts_set_graph,
                target=target_tensor,
            )
            dataset.append(data)
        return dataset

    def get_testing_dataset(self) -> List[TestingData]:
        dataset = []
        mtsa_results = self.testing_mtsa_results
        result_features = self.preprocessor.preprocess(mtsa_results)
        for index, result_feature in enumerate(result_features):
            mtsa_result = mtsa_results[index]
            lts_set_graph = self.get_lts_set_graph(result_feature)
            match self.target_name:
                case "calculation_time":
                    target = mtsa_result.duration_ms
                case "memory_usage":
                    target = mtsa_result.max_memory_usage_kb
                case _:
                    raise ValueError(f"Unknown target name: {self.target_name}")
            target_tensor = torch.tensor(target).to(DEVICE)

            data = TestingData(
                lts_name=mtsa_result.lts,
                lts_set_graph=lts_set_graph,
                target=target_tensor,
            )
            dataset.append(data)
        return dataset
