import datetime
import json
import logging
from typing import List

import numpy as np
import torch.nn.functional as F
import torch.optim

from ml_dcs.config.config import DEVICE
from ml_dcs.domain.ml_gnn import (
    GNNTestingResult,
    GNNTestingResultTask,
    GNNTrainingResult,
    GNNTrainingResultEpoch,
    TestingData,
    TrainingData,
    ValidationData,
)
from ml_dcs.internal.ml.gnn import LTSGNN, GNNDataUtil, LTSRegressionModel
from ml_dcs.internal.mtsa.data_utils import MTSADataUtil
from ml_dcs.internal.signal.signal import Signal, SignalUtil

logger = logging.getLogger(__name__)


# Class that discontinue learning to prevent over-learning
class EarlyStopping:
    def __init__(
        self,
        lts_gnn_model_output_file_path: str,
        regression_model_output_file_path: str,
        patience=20,
        verbose=False,
    ):
        # === args ===
        self.lts_gnn_model_output_file_path = lts_gnn_model_output_file_path
        self.regression_model_output_file_path = regression_model_output_file_path
        # Number of epochs to wait for validation loss to continue to not improve
        self.patience = patience
        self.verbose = verbose

        # === parameters ===
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, lts_gnn_model, regression_model):
        score = -val_loss

        if self.best_score is None:
            # 1st epoch
            self.best_score = score
        elif score < self.best_score:
            # Failure to beat best score
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                # Set Early Stop to True to stop learning
                self.early_stop = True
        else:
            # Successfully updated the best score
            self.best_score = score
            self.save_checkpoint(val_loss, lts_gnn_model, regression_model)
            self.counter = 0

    # executed when best score is updated
    def save_checkpoint(self, val_loss, lts_gnn_model, regression_model):
        if self.verbose:
            logger.info(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    self.val_loss_min, val_loss
                )
            )
        # save model
        torch.save(lts_gnn_model.state_dict(), self.lts_gnn_model_output_file_path)
        torch.save(
            regression_model.state_dict(), self.regression_model_output_file_path
        )
        self.val_loss_min = val_loss

    def is_valid(self):
        return not self.early_stop


class GNNEvaluator:
    ALLOWED_TARGET_NAMES: List[str] = ["calculation_time", "memory_usage"]

    def __init__(
        self,
        input_dir_path: str,
        training_result_output_file_path: str,
        testing_result_output_file_path: str,
        lts_gnn_model_output_file_path: str,
        regression_model_output_file_path: str,
        max_epochs: int,
        target_name: str,
    ):
        # args
        self.input_dir_path = input_dir_path
        self.training_result_output_file_path = training_result_output_file_path
        self.testing_result_output_file_path = testing_result_output_file_path
        self.lts_gnn_model_output_file_path = lts_gnn_model_output_file_path
        self.regression_model_output_file_path = regression_model_output_file_path
        self.max_epochs = max_epochs
        if target_name in self.ALLOWED_TARGET_NAMES:
            self.target_name = target_name
        else:
            raise ValueError(f"Target name {target_name} not allowed")

        # basic parameters
        self.lts_gnn_model = LTSGNN().to(DEVICE)
        self.regression_model = LTSRegressionModel().to(DEVICE)
        self.optimizer = torch.optim.Adam(
            list(self.lts_gnn_model.parameters())
            + list(self.regression_model.parameters()),
            lr=0.001,
        )

    def test(
        self,
        dataset: List[ValidationData | TestingData],
        verbose=False,
        is_validation=False,
    ) -> GNNTestingResult:
        self.lts_gnn_model.eval().to(DEVICE)
        self.regression_model.eval().to(DEVICE)

        task_results = []
        with torch.no_grad():
            for data in dataset:
                started_at = datetime.datetime.now()

                lts_set_embedding = self.lts_gnn_model(data.lts_set_graph)
                prediction = self.regression_model(lts_set_embedding)
                prediction = torch.mean(prediction).to(DEVICE)
                loss = F.mse_loss(prediction, data.target)

                finished_at = datetime.datetime.now()
                task_result = GNNTestingResultTask(
                    lts_name=data.lts_name,
                    loss=loss.item(),
                    actual=data.target.item(),
                    predicted=prediction.item(),
                    started_at=started_at,
                    finished_at=finished_at,
                )
                task_results.append(task_result)
                if verbose:
                    if is_validation:
                        logger.info(
                            "validation result: %s", task_result.model_dump_json()
                        )
                    else:
                        logger.info("testing result: %s", task_result.model_dump_json())

        result = GNNTestingResult(task_results=task_results)
        if verbose:
            if is_validation:
                logger.info("[+] validation_loss_avg: {}".format(result.loss_avg))
            else:
                logger.info("[+] testing_loss_avg: {}".format(result.loss_avg))

        return result

    def train(
        self,
        training_dataset: List[TrainingData],
        validation_dataset: List[ValidationData],
    ) -> GNNTrainingResult:
        started_at = datetime.datetime.now()

        early_stopping = EarlyStopping(
            lts_gnn_model_output_file_path=self.lts_gnn_model_output_file_path,
            regression_model_output_file_path=self.regression_model_output_file_path,
            verbose=True,
        )
        self.lts_gnn_model.to(DEVICE)
        self.regression_model.to(DEVICE)

        epoch_results = []
        for epoch in range(self.max_epochs):
            task_started_at = datetime.datetime.now()

            # ===
            # TRAINING
            # ===
            self.lts_gnn_model.train()
            self.regression_model.train()
            total_loss = 0
            length = len(training_dataset)
            for training_data in training_dataset:
                self.optimizer.zero_grad()
                # get LTS set embedding
                lts_set_embedding = self.lts_gnn_model(training_data.lts_set_graph)
                # regression
                output = self.regression_model(lts_set_embedding)
                # dimension reduction: [-1, 1] -> [1]
                output = torch.mean(output).to(DEVICE)
                # calculate loss
                loss = F.mse_loss(output, training_data.target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            loss_average = total_loss / length

            # ===
            # VALIDATION
            # ===
            validation_result = self.test(
                dataset=validation_dataset, verbose=False, is_validation=True
            )
            early_stopping(
                val_loss=validation_result.loss_avg,
                lts_gnn_model=self.lts_gnn_model,
                regression_model=self.regression_model,
            )

            # ===
            # Post processing
            # ===
            task_finished_at = datetime.datetime.now()
            epoch_result = GNNTrainingResultEpoch(
                training_loss_avg=loss_average,
                validation_loss_avg=validation_result.loss_avg,
                started_at=task_started_at,
                finished_at=task_finished_at,
            )
            epoch_results.append(epoch_result)

            if epoch % 10 == 0:
                logger.info(
                    "training result: %s",
                    json.dumps(
                        {
                            "epoch": f"{epoch + 1}/{self.max_epochs}",
                            "train_loss_avg": str(loss_average),
                            "val_loss_avg": str(validation_result.loss_avg),
                            "val_r_squared": str(validation_result.r_squared),
                        }
                    ),
                )
            else:
                logger.info(
                    "training result: %s",
                    json.dumps(
                        {
                            "epoch": f"{epoch + 1}/{self.max_epochs}",
                            "train_loss_avg": str(loss_average),
                            "val_loss_avg": str(validation_result.loss_avg),
                        }
                    ),
                )

            if not early_stopping.is_valid():
                logger.info("early stopped")
                break

            # ===
            # Manual stopping
            # ===
            stop_manually = SignalUtil.read(Signal.STOP_TRAINING)
            if stop_manually:
                logger.info("manually stopped")
                break

        finished_at = datetime.datetime.now()
        result = GNNTrainingResult(
            epoch_results=epoch_results,
            started_at=started_at,
            finished_at=finished_at,
        )
        return result

    def evaluate(self):
        # load data
        logger.info("loading dataset...")
        mtsa_data_util = MTSADataUtil(self.input_dir_path)
        mtsa_results = mtsa_data_util.get_parsed_data()

        gnn_data_util = GNNDataUtil(mtsa_results, self.target_name)
        training_dataset = gnn_data_util.get_training_dataset()
        validation_dataset = gnn_data_util.get_validation_dataset()
        testing_dataset = gnn_data_util.get_testing_dataset()

        # train
        logger.info("training started")
        training_result = self.train(training_dataset, validation_dataset)
        with open(
            self.training_result_output_file_path, mode="w", encoding="utf-8"
        ) as f:
            f.write(training_result.model_dump_json(indent=2, by_alias=True))
        logger.info("training finished")

        # Get the model before early-stopping
        self.lts_gnn_model.load_state_dict(
            torch.load(self.lts_gnn_model_output_file_path, weights_only=False)
        )
        self.regression_model.load_state_dict(
            torch.load(self.regression_model_output_file_path, weights_only=False)
        )

        # test
        logger.info("testing started")
        testing_result = self.test(dataset=testing_dataset, verbose=True)
        with open(
            self.testing_result_output_file_path, mode="w", encoding="utf-8"
        ) as f:
            f.write(testing_result.model_dump_json(indent=2, by_alias=True))
        logger.info("testing finished")
