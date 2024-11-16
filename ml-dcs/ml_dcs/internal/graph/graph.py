import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from ml_dcs.domain.graph import Graph


class FixedOrderFormatter(ScalarFormatter):
    def __init__(
        self, *args, order_of_mag=0, useOffset=True, useMathText=True, **kwargs
    ):
        self._order_of_mag = order_of_mag
        super().__init__(*args, useOffset=useOffset, useMathText=useMathText, **kwargs)

    def _set_orderOfMagnitude(self):
        self.orderOfMagnitude = self._order_of_mag


class GraphUtil:
    def __init__(self, graph: Graph):
        self.graph = graph

    def _configure(self):
        plt.xlabel(self.graph.x_label)
        plt.ylabel(self.graph.y_label)
        if self.graph.x_lim is not None:
            plt.xlim(*self.graph.x_lim)
        if self.graph.y_lim is not None:
            plt.ylim(*self.graph.y_lim)

        # prediction accuracy plots
        ax1 = plt.gca()
        if self.graph.order_of_mag is not None:
            ax1.xaxis.set_major_formatter(
                FixedOrderFormatter(order_of_mag=self.graph.order_of_mag)
            )
            ax1.yaxis.set_major_formatter(
                FixedOrderFormatter(order_of_mag=self.graph.order_of_mag)
            )
            ax1.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

        ax1.scatter(
            self.graph.data.x,
            self.graph.data.y,
            c="white",
            edgecolors="blue",
            s=50,
        )

        # y=x plots
        x = np.linspace(-(5**10), 10**10, 100)
        y = np.linspace(-(5**10), 10**10, 100)
        ax2 = plt.gca()
        ax2.plot(x, y, linestyle="--", color="red")

    def show(self):
        self._configure()
        plt.show()

    def save(self):
        self._configure()
        plt.savefig(self.graph.output_path)
