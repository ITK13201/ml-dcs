import os
import random
from logging import config

import numpy as np
import torch
from matplotlib import pyplot as plt

# ===
# LOGGING
# ===
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[%(levelname)s] %(asctime)s %(name)s:%(lineno)s %(funcName)s %(module)s %(process)d %(thread)d: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "verbose",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "verbose",
            "filename": "ml-dcs.log",
        },
    },
    "loggers": {
        "__main__": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},
    },
}

config.dictConfig(LOGGING)

# ===
# MATPLOTLIB
# ===
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "black"

# ===
# PyTorch
# ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch Device: {}".format(str(DEVICE)))

# ===
# SIGNAL
# ===
DEFAULT_SIGNAL_DIR = os.path.join("tmp", "signals")

# ===
# ML Random Seed
# ===
DEFAULT_RANDOM_SEED = 42
random.seed(DEFAULT_RANDOM_SEED)
np.random.seed(DEFAULT_RANDOM_SEED)
torch.manual_seed(DEFAULT_RANDOM_SEED)
torch.cuda.manual_seed(DEFAULT_RANDOM_SEED)
