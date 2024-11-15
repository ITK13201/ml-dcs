import os.path
from enum import Enum

from ml_dcs.config.config import DEFAULT_SIGNAL_DIR


class Signal(Enum):
    STOP_TRAINING = "STOP_TRAINING"


class SignalUtil:
    signal_dir = DEFAULT_SIGNAL_DIR

    @classmethod
    def set_signal_dir(cls, signal_dir: str):
        cls.signal_dir = signal_dir

    @classmethod
    def _get_signal_file_path(cls, signal: Signal) -> str:
        return os.path.join(cls.signal_dir, "{}.signal".format(signal.value))

    # read signal file (e.g., STOP_TRAINING.signal)
    @classmethod
    def read(cls, signal: Signal) -> bool:
        file_path = cls._get_signal_file_path(signal)
        if os.path.isfile(file_path):
            os.remove(file_path)
            return True
        else:
            return False
