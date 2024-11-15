import logging

from ml_dcs.cmd.base import BaseCommand

logger = logging.getLogger(__name__)


class PredictCalculationTimeCommand(BaseCommand):
    name = "predict_calculation_time"
    help = "Predict calculation time"
