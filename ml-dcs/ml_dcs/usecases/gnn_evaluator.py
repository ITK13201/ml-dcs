import logging
from typing import List

import torch.optim

from ml_dcs.config.config import DEVICE
from ml_dcs.internal.ml.gnn_complex import LTSGNN, LTSRegressionModel, GNNDataUtil
from ml_dcs.internal.mtsa.data_utils import MTSADataUtil

import torch.nn.functional as F

logger = logging.getLogger(__name__)

class GNNEvaluator:
    ALLOWED_TARGET_NAMES: List[str] = [
        "calculation_time",
        "memory_usage"
    ]

    def __init__(self, input_dir_path: str, epochs: int, target_name: str):
        # args
        self.input_dir_path = input_dir_path
        self.epochs = epochs
        if target_name in self.ALLOWED_TARGET_NAMES:
            self.target_name = target_name
        else:
            raise ValueError(f"Target name {target_name} not allowed")

        # basic parameters
        self.lts_gnn_model = LTSGNN().to(DEVICE)
        self.regression_model = LTSRegressionModel().to(DEVICE)
        self.optimizer = torch.optim.Adam(list(self.lts_gnn_model.parameters()) + list(self.regression_model.parameters()), lr=0.001)

    def train(self, training_dataset):
        for epoch in range(self.epochs):
            self.lts_gnn_model.train().to(DEVICE)
            self.regression_model.train().to(DEVICE)
            total_loss = 0
            for lts_set_graph, target in training_dataset:
                self.optimizer.zero_grad()
                # get LTS set embedding
                lts_set_embedding = self.lts_gnn_model(lts_set_graph)
                # regression
                output = self.regression_model(lts_set_embedding)
                # dimension reduction: [-1, 1] -> [1]
                output = torch.mean(output).to(DEVICE)

                # calculation loss
                loss = F.mse_loss(output, target)
                loss.backward()
                self.optimizer.step()
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

                total_loss += loss.item()
            logger.info(f"[+] Epoch {epoch + 1}/{self.epochs}, Total Loss: {total_loss}")

    def test(self, testing_dataset):
        self.lts_gnn_model.eval()
        self.regression_model.eval()
        with torch.no_grad():
            total_loss = 0
            for lts_set_graph, target in testing_dataset:
                prediction = self.lts_gnn_model(lts_set_graph)
                loss = F.mse_loss(prediction, target)
                logger.info(f"Loss: {loss.item()}")

                total_loss += loss
            logger.info(f"Total Loss: {total_loss}")


    def evaluate(self):
        mtsa_data_util = MTSADataUtil(self.input_dir_path)
        mtsa_results = mtsa_data_util.get_parsed_data()

        gnn_data_util = GNNDataUtil(mtsa_results, self.target_name)
        # train
        self.train(gnn_data_util.get_training_dataset())
        # test
        self.test(gnn_data_util.get_testing_dataset())
