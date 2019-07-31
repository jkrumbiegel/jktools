from jktools.geometry import read_svg_paths, convert_svg_path_to_mpl
import jktools
import os
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from collections import OrderedDict
import numpy as np


HEAD_SVG_PATH = os.path.join(os.path.dirname(jktools.__file__), 'plot/files/head-top-down.svg')
svg_paths = read_svg_paths(HEAD_SVG_PATH)
paths = OrderedDict((key, convert_svg_path_to_mpl(value)) for key, value in svg_paths.items())


def head(center, width, height=None, angle=0, degrees=True, **kwargs):
    outline_vertices = paths['outline'].vertices
    min_extent = np.nanmin(outline_vertices, axis=0)
    max_extent = np.nanmax(outline_vertices, axis=0)
    headcenter = (min_extent + max_extent) / 2
    headwidth, headheight = max_extent - min_extent
    scalar_width = width / headwidth
    scalar_height = scalar_width if height is None else height / headheight

    angle = np.deg2rad(angle) if degrees else angle

    patches = OrderedDict()
    for key, path in paths.items():
        new_vertices = ((path.vertices - headcenter) # move to origin
                        * np.array([scalar_width, scalar_height])# scale
                        @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) # rotate
                        + center) # move to position
        patches[key] = PathPatch(Path(new_vertices, codes=path.codes), **kwargs)

    ax = kwargs.pop('ax', plt.gca())

    for name, patch in patches.items():
        ax.add_patch(patch)
    return patches
