import numpy as np
import matplotlib.pyplot as plt


class FuncPlot:

    def __init__(self, func, n=100, margined=False, ax=None, **kwargs):
        self.func = func
        self.margined = margined
        self.n = n
        self.ax = ax if ax is not None else plt.gca()
        (self.line,) = ax.plot([], [], **kwargs)

        self.x_callback_id = self.ax.callbacks.connect('xlim_changed', self.limits_updated)
        self.y_callback_id = self.ax.callbacks.connect('ylim_changed', self.limits_updated)
        self.limits_updated(self.ax)

    def limits_updated(self, axis):
        xlims = axis.get_xlim()
        if self.margined:
            xmargin, ymargin = axis.margins()
            xrange = xlims[1] - xlims[0]
            margined_xlims = xlims[0] + xmargin * xrange, xlims[1] - xmargin * xrange
            xx = np.linspace(*margined_xlims, self.n)
        else:
            xx = np.linspace(*xlims, self.n)
        yy = self.func(xx)

        self.line.set_data(xx, yy)
