from logging import config

from matplotlib import pyplot as plt

# ===
# LOGGING
# ===
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[%(levelname)s] %(asctime)s %(module)s %(process)d %(thread)d %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "verbose",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "__main__": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "": {"level": "INFO", "handlers": ["console"], "propagate": False},
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
