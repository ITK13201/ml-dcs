import logging

logger = logging.getLogger(__name__)


class PredictMemoryUsageGNNExecutor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mode = "gnn"

    def execute(self):
        logger.info("mode: %s", self.mode)
