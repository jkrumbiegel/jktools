import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
import numpy as np


def circular_arrow(center, radius, angle, theta2, color='k', ax=None, zorder=1, lw=1, tipwidth=3, start_tip=False, end_tip=True):

    if ax is None:
        ax = plt.gca()

    arc = Arc(center, radius * 2, radius * 2, angle=angle,
              theta1=0, theta2=theta2, capstyle='round', linestyle='-', lw=lw, color=color, zorder=zorder)

    ax.add_patch(arc)

    if end_tip:
        end_x = center[0] + radius * np.cos(np.deg2rad(theta2 + angle))
        end_y = center[1] + radius * np.sin(np.deg2rad(theta2 + angle))

        ax.add_patch(                    #Create triangle as arrow head
            RegularPolygon(
                (end_x, end_y),            # (x,y)
                3,                       # number of vertices
                tipwidth / 2,                # radius
                np.deg2rad(angle + theta2),     # orientation
                color=color,
                zorder=zorder
            ))

    if start_tip:
        start_x = center[0] + radius * np.cos(np.deg2rad(angle))
        start_y = center[1] + radius * np.sin(np.deg2rad(angle))

        ax.add_patch(  # Create triangle as arrow head
            RegularPolygon(
                (start_x, start_y),  # (x,y)
                3,  # number of vertices
                tipwidth / 2,  # radius
                np.deg2rad(angle + 180),  # orientation
                color=color,
                zorder=zorder
            ))
