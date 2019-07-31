import matplotlib.transforms as trans
import matplotlib.pyplot as plt


def offset_text(text, x, y, x_offset, y_offset, transform='axis', fig=None, units='points', **kwargs):

    ax = kwargs.pop('ax', plt.gca())

    if fig is None:
        fig = ax.figure

    if transform == 'axis':
        transform = ax.transAxes
    elif transform == 'data':
        transform = ax.transData

    offset_transform = trans.offset_copy(transform, fig, x_offset, y_offset, units=units)

    ax.text(x, y, text, transform=offset_transform, **kwargs)
