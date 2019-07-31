from jktools.plot import offset_text
import matplotlib.pyplot as plt


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)

for ax, t in zip(fig.axes, ['A', 'B', 'C']):
    offset_text(t, 0, 1, -40, 0, ax=ax, fontsize=20, fontweight='bold', va='top')

plt.show()