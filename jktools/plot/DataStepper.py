from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import sys

class DataStepper(QtWidgets.QWidget):
    def __init__(self, plot_function, index):
        super(DataStepper, self).__init__(None)
        self.plot_function = plot_function
        # takes forward, backward, or index
        self.index = index
        self.i = 0

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.prev_button = QtWidgets.QPushButton('Previous')
        self.prev_button.clicked.connect(lambda: [self.prev_index(), self.draw_plot()])

        self.next_button = QtWidgets.QPushButton('Next')
        self.next_button.clicked.connect(lambda: [self.next_index(), self.draw_plot()])

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        button_sublayout = QtWidgets.QHBoxLayout()
        button_sublayout.addWidget(self.prev_button)
        button_sublayout.addWidget(self.next_button)
        layout.addLayout(button_sublayout)

        self.setLayout(layout)
        self.draw_plot()
        self.show()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.next_button.animateClick(100)
        elif event.key() == QtCore.Qt.Key_Left:
            self.prev_button.animateClick(100)

    def prev_index(self):
        self.i = (self.i - 1) % len(self.index)

    def next_index(self):
        self.i = (self.i + 1) % len(self.index)

    def draw_plot(self):
        self.figure.clear()
        self.plot_function(self.figure, self.index[self.i])
        self.figure.canvas.draw()