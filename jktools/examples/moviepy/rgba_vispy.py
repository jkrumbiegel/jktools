import jktools.moviepy as jmpy
import numpy as np
import moviepy.editor as mpy
from vispy import app, scene
from vispy.gloo.util import _screenshot

size_pixels = 600, 400
size_pixels_half = 300, 200
duration = 5

canvas = scene.SceneCanvas(keys='interactive', size=size_pixels_half, resizable=False, bgcolor=(0, 0, 0, 0))
view = canvas.central_widget.add_view()

view.camera = 'turntable'
view.camera.fov = 40
view.camera.distance = 10

xx, yy = np.arange(-1, 1, .02), np.arange(-1, 1, .02)
X, Y = np.meshgrid(xx, yy)
R = np.sqrt(X ** 2 + Y ** 2)
Z = lambda t: 0.1 * np.sin(10 * R - 2 * np.pi * t)
surface = scene.visuals.SurfacePlot(x=xx - 0.1, y=yy + 0.2, z=Z(0),
                                    shading='smooth', color=(0.5, 0.5, 1, 1))
view.add(surface)
canvas.show()


def make_frame(t):
    surface.set_data(z=Z(t))  # Update the mathematical surface
    canvas.on_draw(None)  # Update the image on Vispy's canvas
    return _screenshot((0, 0, size_pixels[0], size_pixels[1]))


vispy_clip = jmpy.RGBAVideoClip(make_frame, duration=duration)
bg_clip = mpy.ColorClip(size_pixels, color=(255, 0, 0), duration=duration)

composite = mpy.CompositeVideoClip([bg_clip, vispy_clip])
composite.preview()
