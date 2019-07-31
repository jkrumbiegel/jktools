import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt


def eye(
        center,
        fixation,
        eye_radius,
        fixation_radius,
        iris_percentage=0.8,
        pupil_percentage=0.4,
        ax=None,
        eye_kws=None,
        fix_kws=None,
        iris_kws=None,
        pupil_kws=None,
        tangent_kws=None):
    if not ax:
        ax = plt.gca()

    center = np.array(center) if not isinstance(center, np.ndarray) else center
    fixation = np.array(fixation) if not isinstance(fixation, np.ndarray) else fixation
    fixation_to_center = (fixation - center)
    fixation_to_center_distance = np.linalg.norm(fixation_to_center)
    direction_to_fixation = fixation_to_center / fixation_to_center_distance
    pupil_center = center + direction_to_fixation * eye_radius

    a = np.arcsin(fixation_radius / fixation_to_center_distance)
    b = np.arctan2(fixation_to_center[1], fixation_to_center[0])

    t_1 = b - a
    tangent_1 = np.array([np.sin(t_1), -np.cos(t_1)]) * fixation_radius + fixation
    t_2 = b + a
    tangent_2 = np.array([-np.sin(t_2), np.cos(t_2)]) * fixation_radius + fixation

    ax.plot([center[0], tangent_1[0]], [center[1], tangent_1[1]], **(tangent_kws if tangent_kws is not None else {}),
            zorder=3)
    ax.plot([center[0], tangent_2[0]], [center[1], tangent_2[1]], **(tangent_kws if tangent_kws is not None else {}),
            zorder=3)

    eye_patch = patches.Circle(
        center,
        eye_radius,
        zorder=4,
        **(dict((key, value) for key, value in eye_kws.items() if key != 'edgecolor') if eye_kws is not None else {}),
        edgecolor=(0, 0, 0, 0)
    )
    ax.add_patch(eye_patch)

    eye_patch_upper = patches.Circle(
        center,
        eye_radius,
        zorder=10,
        **(dict((key, value) for key, value in eye_kws.items() if key != 'facecolor') if eye_kws is not None else {}),
        facecolor=(0, 0, 0, 0)
    )
    ax.add_patch(eye_patch_upper)

    iris_patch = patches.Circle(
        pupil_center,
        eye_radius * iris_percentage,
        zorder=4,
        **(iris_kws if iris_kws is not None else {})
    )
    iris_patch.set_clip_path(eye_patch)
    ax.add_patch(iris_patch)

    pupil_patch = patches.Circle(
        pupil_center,
        eye_radius * pupil_percentage,
        zorder=4,
        **(pupil_kws if pupil_kws is not None else {})
    )
    pupil_patch.set_clip_path(eye_patch)
    ax.add_patch(pupil_patch)

    #     reflex_patch = patches.Circle(
    #         center + np.array([1, 1]) / np.sqrt(2) * eye_radius * 0.83,
    #         eye_radius / 10,
    #         zorder=9,
    #         facecolor=(1, 1, 1, 0.8),
    #         edgecolor=(0, 0, 0, 0)
    #     )
    #     ax.add_patch(reflex_patch)

    fixation_patch = patches.Circle(
        fixation,
        fixation_radius,
        zorder=4,
        **(fix_kws if fix_kws is not None else {})
    )
    ax.add_patch(fixation_patch)