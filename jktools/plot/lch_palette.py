from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color
import numpy as np


def lch_palette(n, l=0.7, h=0, s=0.7):

    lch_colors = [LCHabColor(l * 100, s * 100, ((h * 360) + hue) % 360) for hue in np.linspace(0, 360, n + 1)[:-1]]
    srgb_colors = [convert_color(lch, sRGBColor) for lch in lch_colors]
    return [(srgb.clamped_rgb_r, srgb.clamped_rgb_g, srgb.clamped_rgb_b) for srgb in srgb_colors]
