import matplotlib.pyplot as plt


def get_clean_fig_ax(size_pixels, xlims=None, ylims=None, dpi=200, facecolor=(0, 0, 0, 0)):
    fig = plt.figure(figsize=(size_pixels[0] / dpi, size_pixels[1] / dpi), dpi=dpi, facecolor=facecolor)
    ax = fig.add_subplot(111)
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
    # stretch axis to figure borders
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    # remove axis markings
    plt.axis('off')
    return fig, ax
