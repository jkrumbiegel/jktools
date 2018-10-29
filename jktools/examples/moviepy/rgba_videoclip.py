import matplotlib
matplotlib.use('Agg')
import jktools.moviepy as jmpy
import numpy as np
import moviepy.editor as mpy

size_pixels = 600, 400
duration = 10
fig, ax = jmpy.get_clean_fig_ax(size_pixels)

xx = np.linspace(0, 2 * np.pi, 200)
yy = np.sin(xx)

line, = ax.plot(xx, yy)

def make_frame(t):
    new_yy = np.sin(xx + t)
    line.set_data(xx, new_yy)
    fig.canvas.draw()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(size_pixels[1], size_pixels[0], 4)

def make_frame2(t):
    new_yy = np.sin(xx + t + np.pi)
    line.set_data(xx, new_yy)
    fig.canvas.draw()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(size_pixels[1], size_pixels[0], 4)

plot_clip = jmpy.RGBAVideoClip(make_frame, duration=duration)
plot_clip2 = jmpy.RGBAVideoClip(make_frame2, duration=duration)
bg_clip = mpy.ColorClip(size_pixels, color=(255, 0, 0), duration=duration)

composite = mpy.CompositeVideoClip([bg_clip, plot_clip, plot_clip2])
composite.preview()