import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from ml_dcs.domain.graph import Graph, Graph2


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


class Graph2Util:
    def __init__(self, graph: Graph2):
        self.graph = graph
        self.early_stopping_threshold = 10

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
            ax1.yaxis.set_major_formatter(
                FixedOrderFormatter(order_of_mag=self.graph.order_of_mag)
            )
            ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # y1
        ax1.plot(
            self.graph.data.x,
            self.graph.data.y1,
            color="blue",
            label=self.graph.y1_legend,
        )
        # y2
        ax1.plot(
            self.graph.data.x,
            self.graph.data.y2,
            color="orange",
            label=self.graph.y2_legend,
        )

        # best score plots
        ax2 = plt.gca()
        x = np.linspace(
            max(self.graph.data.x) - self.early_stopping_threshold,
            max(self.graph.data.x) - self.early_stopping_threshold,
            100,
        )
        y = np.linspace(-(10**15), 10**15, 100)
        ax2.plot(x, y, linestyle="--", color="red")

        plt.legend(loc="upper right")

    def show(self):
        self._configure()
        plt.show()

    def save(self):
        self._configure()
        plt.savefig(self.graph.output_path)
