from . import offset_text
import string
import matplotlib.pyplot as plt


def subplot_letters(
        figure=None,
        letters=list(string.ascii_uppercase),
        fontsize=20,
        fontweight='bold',
        dx=-40,
        dy=0,
        x=0,
        y=1,
        va='top',
        transform='axis',
        **kwargs):

    if figure is None:
        figure = plt.gcf()

    for ax, t in zip(figure.axes, letters):
        offset_text(
            t, x, y, dx, dy, ax=ax, fontsize=fontsize, fontweight=fontweight, va=va, transform=transform, **kwargs)
