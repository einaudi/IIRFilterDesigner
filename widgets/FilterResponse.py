# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTabWidget
from widgets.PlotCanvas import PlotCanvas


class tabTransient(QWidget):

    def __init__(self):

        super().__init__()

        # widgets
        self._canvas = PlotCanvas(xLabel="Time [s]", yLabel="Response")
        self._canvas.set_style()

        # layout
        layout = QVBoxLayout()
        layout.addWidget(self._canvas)

        self.setLayout(layout)

class tabFrequency(QWidget):

    def __init__(self):

        super().__init__()

        # widgets
        self._freq = QLineEdit()
        self._canvas = PlotCanvas(xLabel="Time [s]", yLabel="Response")
        self._canvas.set_style()

        # layout
        settingsLayout = QHBoxLayout()
        settingsLayout.addWidget(QLabel("Frequency [Hz]"))
        settingsLayout.addWidget(self._freq)

        layout = QVBoxLayout()
        layout.addLayout(settingsLayout)
        layout.addWidget(self._canvas)

        self.setLayout(layout)

class FilterResponse(QTabWidget):
    
    def __init__(self):

        super().__init__()

        # tabs
        self._tabTransient = tabTransient()
        self._tabFrequency = tabFrequency()

        # layout
        self.addTab(self._tabTransient, "Transient")
        self.addTab(self._tabFrequency, "Frequency")