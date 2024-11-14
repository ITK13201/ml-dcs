import logging
from typing import Iterator, List, Tuple

import torch
import torch.nn.functional as F
from hypothesis import target
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from ml_dcs.config.config import DEVICE
from ml_dcs.domain.mtsa import MTSAResult

logger = logging.getLogger(__name__)

LTS_EMBEDDING_SIZE = 8
DEFAULT_RANDOM_STATE = 42


class LTSGNN(torch.nn.Module):
    INPUT_CHANNELS = 5
    HIDDEN_CHANNELS = 16
    OUTPUT_CHANNELS = LTS_EMBEDDING_SIZE

    def __init__(self):
        super(LTSGNN, self).__init__()
        self.conv1 = GCNConv(self.INPUT_CHANNELS, self.HIDDEN_CHANNELS).to(DEVICE)
        self.conv2 = GCNConv(self.HIDDEN_CHANNELS, self.OUTPUT_CHANNELS).to(DEVICE)

    def forward(self, data: Data):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
        )

        # 1st graph convolution
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        # 2nd graphh convolution
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)

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

    def _split_mtsa_results(self, mtsa_results: List[MTSAResult]):
        training_results, testing_results = train_test_split(
            mtsa_results, test_size=0.2, random_state=DEFAULT_RANDOM_STATE
        )
        return training_results, testing_results

    def __init__(self, mtsa_results: List[MTSAResult], target_name: str):
        self.mtsa_results = mtsa_results
        if target_name in self.ALLOWED_TARGET_NAMES:
            self.target_name = target_name
        else:
            raise ValueError(f"Target name {target_name} not allowed")
        self.training_mtsa_results, self.testing_mtsa_results = (
            self._split_mtsa_results(self.mtsa_results)
        )

    def get_lts_graph(
        self,
        lts: List[List[int]],
        lts_number: int,
        number_of_states: int,
        is_environment: bool,
    ) -> Data:
        error_state_number = number_of_states - 1

        edge_index = []
        edge_weight = []
        for transition in lts:
            (src_state_number, dst_state_number, action_weight, is_controllable) = (
                transition
            )

            if dst_state_number != -1:
                edge_index.append([src_state_number, dst_state_number])
            else:
                # if dst is error state
                edge_index.append([src_state_number, error_state_number])
            edge_weight.append(action_weight)
        edge_index_tensor = (
            torch.tensor(edge_index, dtype=torch.int).t().contiguous().to(DEVICE)
        )
        edge_weight_tensor = (
            torch.tensor(edge_weight, dtype=torch.float).to(DEVICE).to(DEVICE)
        )

        # node_feature: (is_environment, lts_number, state_number, is_error_state, is_start_state)
        if is_environment:
            # environment
            node_features = []
            for index in range(number_of_states):
                if index == 0:
                    # start state
                    node_features.append([1, lts_number, index, 0, 1])
                else:
                    node_features.append([1, lts_number, index, 0, 0])
        else:
            # requirement
            node_features = []
            for index in range(number_of_states):
                if index == 0:
                    # start state
                    node_features.append([0, lts_number, index, 0, 1])
                elif index == number_of_states - 1:
                    # error state
                    node_features.append([0, lts_number, index, 1, 0])
                else:
                    node_features.append([0, lts_number, index, 0, 1])
        node_features_tensor = torch.tensor(node_features, dtype=torch.float).to(DEVICE)

        lts_embedding = Data(
            x=node_features_tensor,
            edge_index=edge_index_tensor,
            edge_weight=edge_weight_tensor,
        ).to(DEVICE)

        return lts_embedding

    def get_lts_set_graph(self, mtsa_result: MTSAResult):
        # initialize
        mtsa_result.initialize_quantified_structures()

        lts_embeddings = []
        cnt = 0
        for environment in mtsa_result.initial_models.environments:
            lts_embeddings.append(
                self.get_lts_graph(
                    lts=environment.quantified_structure,
                    lts_number=cnt,
                    number_of_states=environment.number_of_states,
                    is_environment=True,
                )
            )
        for requirement in mtsa_result.initial_models.requirements:
            lts_embeddings.append(
                self.get_lts_graph(
                    lts=requirement.quantified_structure,
                    lts_number=cnt,
                    number_of_states=requirement.number_of_states,
                    is_environment=False,
                )
            )

        batch = Batch.from_data_list(lts_embeddings)
        return batch

    def get_training_dataset(self):
        dataset = []
        for mtsa_result in self.training_mtsa_results:
            mtsa_result: MTSAResult
            lts_set_graph = self.get_lts_set_graph(mtsa_result)
            match self.target_name:
                case "calculation_time":
                    target = mtsa_result.duration_ms
                case "memory_usage":
                    target = mtsa_result.max_memory_usage_kib
                case _:
                    raise ValueError(f"Unknown target name: {self.target_name}")
            target_tensor = torch.tensor(target).to(DEVICE)
            dataset.append([lts_set_graph, target_tensor])
        return dataset

    def get_testing_dataset(self):
        dataset = []
        for mtsa_result in self.testing_mtsa_results:
            mtsa_result: MTSAResult
            lts_set_graph = self.get_lts_set_graph(mtsa_result)
            match self.target_name:
                case "calculation_time":
                    target = mtsa_result.duration_ms
                case "memory_usage":
                    target = mtsa_result.max_memory_usage_kib
                case _:
                    raise ValueError(f"Unknown target name: {self.target_name}")
            target_tensor = torch.tensor(target).to(DEVICE)
            dataset.append([lts_set_graph, target_tensor])
        return dataset
