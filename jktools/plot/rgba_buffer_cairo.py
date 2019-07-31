import numpy as np
from matplotlib.cbook import _premultiplied_argb32_to_unmultiplied_rgba8888


def rgba_buffer_cairo(fig, size):
    buffer = fig.canvas._get_printed_image_surface().get_data()
    return _premultiplied_argb32_to_unmultiplied_rgba8888(
        np.frombuffer(buffer, np.uint8).reshape((size[1], size[0], 4)))
