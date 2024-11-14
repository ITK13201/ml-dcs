import itertools
import json
import logging
from typing import Iterator, List

import torch
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

from ml_dcs.config.config import DEVICE
from ml_dcs.domain.mtsa import MTSAResult

logger = logging.getLogger(__name__)

class GNNDataUtil:
    ALLOWED_TARGET_NAMES: List[str] = [
        "calculation_time",
        "memory_usage"
    ]
    def __init__(self, mtsa_results: Iterator[MTSAResult], target_name: str):
        self.mtsa_results = mtsa_results
        if target_name in self.ALLOWED_TARGET_NAMES:
            self.target_name = target_name
        else:
            raise ValueError(f"Target name {target_name} not allowed")


    def get_lts_set_graph(self, mtsa_result: MTSAResult) -> Data:
        match self.target_name:
            case "calculation_time":
                target = mtsa_result.duration_ms
            case "memory_usage":
                target = mtsa_result.max_memory_usage_kib
            case _:
                raise ValueError(f"Unknown target name: {self.target_name}")

        # initialize
        mtsa_result.initialize_quantified_structures()

        node_features = []
        max_node_size = -1
        environment_count = 0
        for environment in mtsa_result.initial_models.environments:
            node_feature = torch.tensor(environment.quantified_structure, dtype=torch.float).to(DEVICE)
            max_node_size = max(max_node_size, len(node_feature))
            node_features.append(node_feature)
            environment_count += 1
        requirement_count = environment_count
        for requirement in mtsa_result.initial_models.requirements:
            node_feature = torch.tensor(requirement.quantified_structure, dtype=torch.float).to(DEVICE)
            max_node_size = max(max_node_size, len(node_feature))
            node_features.append(node_feature)
            requirement_count += 1

        padded_node_features = []
        for node_feature in node_features:
            padded_node_feature = torch.cat([node_feature, torch.zeros(max_node_size - len(node_feature), 4).to(DEVICE)], dim=0).to(DEVICE)
            padded_node_features.append(padded_node_feature)
        padded_node_features_tensor = torch.stack(padded_node_features, dim=0).to(DEVICE)

        environment_node_numbers = list(range(0, environment_count))
        requirement_node_numbers = list(range(environment_count, requirement_count))

        edge_index = []
        edge_weight = []
        environment_node_combinations = itertools.combinations(environment_node_numbers, 2)
        for environment_node_combination in environment_node_combinations:
            # undirected graph
            edge_index.append(environment_node_combination)
            edge_weight.append(0)
            edge_index.append(reversed(environment_node_combination))
            edge_weight.append(0)
        requirement_node_combinations = itertools.combinations(requirement_node_numbers, 2)
        for requirement_node_combination in requirement_node_combinations:
            # undirected graph
            edge_index.append(requirement_node_combination)
            edge_weight.append(1)
            edge_index.append(reversed(requirement_node_combination))
            edge_weight.append(1)
        product_of_environments_and_requirements = itertools.product(environment_node_numbers, requirement_node_numbers)
        for combination in product_of_environments_and_requirements:
            # undirected graph
            edge_index.append(combination)
            edge_weight.append(2)
            edge_index.append(reversed(combination))
            edge_weight.append(2)

        target_tensor = torch.tensor([target], dtype=torch.float).to(DEVICE)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.int).t().contiguous().to(DEVICE)
        edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.float).to(DEVICE)

        lts_set_graph = Data(x=padded_node_features_tensor, edge_index=edge_index_tensor, edge_weight=edge_weight_tensor, y=target_tensor).to(DEVICE)
        return lts_set_graph

    def get_dataloader(self) -> DataLoader:
        dataset = []
        for mtsa_result in self.mtsa_results:
            lts_set_graph = self.get_lts_set_graph(mtsa_result).to(DEVICE)
            dataset.append(lts_set_graph)
        # loader = DataLoader(dataset, batch_size=1, shuffle=True)
        loader = DataLoader(dataset, batch_size=1)
        return loader


class GCNRegression(torch.nn.Module):
    HIDDEN_CHANNELS = 16
    # output is scalar (calculation_time or memory_usage)
    OUTPUT_CHANNELS = 1

    def __init__(self, in_channels: int):
        super(GCNRegression, self).__init__()
        self.conv1 = GCNConv(in_channels, self.HIDDEN_CHANNELS)
        self.conv2 = GCNConv(self.HIDDEN_CHANNELS, self.HIDDEN_CHANNELS)
        self.fc1 = torch.nn.Linear(self.HIDDEN_CHANNELS, self.OUTPUT_CHANNELS)

    def forward(self, data: Data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # 1st graph convolution
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        # 2nd graph convolution
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)

        # pooling
        x = global_mean_pool(x, batch)

        # output
        x = self.fc1(x)
        return x


class GNNTrainer:
    def __init__(self, dataloader: DataLoader, epochs: int):
        self.dataloader = dataloader
        self.epochs = epochs

        # basic parameters
        self.model = GCNRegression(in_channels=4).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def train(self) -> GCNRegression:
        for epoch in range(self.epochs):
            self.model.train().to(DEVICE)
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch.y)
                loss.backward()
                self.optimizer.step()
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

        return self.model

    def evaluate(self, data: Data):
        print("evaluation")
        self.model.eval()
        print("evaluation")
        # with torch.no_grad():
        prediction = self.model(data)
        print("evaluation")
        loss = F.mse_loss(prediction, data.y)
        print("evaluation")
        logger.info(f"Loss: {loss.item()}")
        print(prediction)
        print(data.y)

        return 1
