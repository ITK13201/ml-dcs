from matplotlib.ticker import ScalarFormatter


class FixedOrderFormatter(ScalarFormatter):
    def __init__(
        self, *args, order_of_mag=0, useOffset=True, useMathText=True, **kwargs
    ):
        self._order_of_mag = order_of_mag
        super().__init__(*args, useOffset=useOffset, useMathText=useMathText, **kwargs)

    def _set_orderOfMagnitude(self):
        self.orderOfMagnitude = self._order_of_mag
