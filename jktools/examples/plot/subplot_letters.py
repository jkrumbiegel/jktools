import matplotlib.pyplot as plt
from jktools.plot import subplot_letters


fig, axes = plt.subplots(3, 4)
subplot_letters(fig)
fig.subplots_adjust(wspace=0.3)

plt.show()