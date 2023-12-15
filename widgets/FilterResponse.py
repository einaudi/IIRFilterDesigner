# -*- coding: utf-8 -*-

import numpy as np

from src.IIR import IIRFilter

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTabWidget, QPushButton
from widgets.PlotCanvas import PlotCanvas
from widgets.Dialogs import dialogWarning


class tabTransient(QWidget):

    def __init__(self):

        super().__init__()

        # variables
        self._fb_coefs = []
        self._ff_coefs = []
        self._freqSampling = 1
        self._flag_filter = False

        # widgets
        self._canvas = PlotCanvas(xLabel="Time [s]", yLabel="Response")
        self._canvas.set_style()
        self._btnCalculate = QPushButton('Calculate')
        self._editTime = QLineEdit()
        self._editTime.setText('1e-2')

        # layout
        settingsBox = QHBoxLayout()
        settingsBox.addWidget(QLabel('Time [s]'))
        settingsBox.addWidget(self._editTime)
        settingsBox.addWidget(self._btnCalculate)

        layout = QVBoxLayout()
        layout.addLayout(settingsBox)
        layout.addWidget(self._canvas)

        self.setLayout(layout)

        # UI
        self._btnCalculate.clicked.connect(self._calcResponse)

    def getCoefs(self, fb_coefs, ff_coefs, freqSampling):

        self._fb_coefs = fb_coefs
        self._ff_coefs = ff_coefs
        self._freqSampling = freqSampling
        self._flag_filter = True

    def _calcResponse(self):

        if not self._flag_filter:
            dialogWarning('Calculate filter parameters first!')
            return False

        try:
            T = float(self._editTime.text())
        except ValueError:
            dialogWarning('Transient time invalid!')
            return False

        if T <= 0:
            dialogWarning('Transient time must be positive!')
            return False

        # calculate transient response
        N = int(1.1*T*self._freqSampling)
        ts = np.linspace(-0.1*T, T, N)
        signal = np.zeros(N)
        signal = np.where(
            ts >= 0,
            1,
            0
        )

        filt = IIRFilter(self._ff_coefs, self._fb_coefs)
        ys_filtered = []
        for i in range(N):
            ys_filtered.append(filt.update(signal[i]))

        # Plot
        self._canvas.prepare_axes()
        self._canvas.set_style()
        self._canvas.plot(
            ts,
            signal,
            label='input'
        )
        self._canvas.plot(
            ts,
            ys_filtered,
            label='filtered'
        )
        self._canvas.add_legend()
        self._canvas.refresh()

        return True


class tabFrequency(QWidget):

    def __init__(self):

        super().__init__()

        # variables
        self._fb_coefs = []
        self._ff_coefs = []
        self._freqSampling = 1
        self._flag_filter = False

        # widgets
        self._canvas = PlotCanvas(xLabel="Time [s]", yLabel="Response")
        self._canvas.set_style()
        self._btnCalculate = QPushButton('Calculate')
        self.editFrequency = QLineEdit()
        self.editFrequency.setText('1e2')

        # layout
        settingsBox = QHBoxLayout()
        settingsBox.addWidget(QLabel('Frequency [Hz]'))
        settingsBox.addWidget(self.editFrequency)
        settingsBox.addWidget(self._btnCalculate)

        layout = QVBoxLayout()
        layout.addLayout(settingsBox)
        layout.addWidget(self._canvas)

        self.setLayout(layout)

        # UI
        self._btnCalculate.clicked.connect(self._calcResponse)

    def getCoefs(self, fb_coefs, ff_coefs, freqSampling):

        self._fb_coefs = fb_coefs
        self._ff_coefs = ff_coefs
        self._freqSampling = freqSampling
        self._flag_filter = True

    def _calcResponse(self):

        if not self._flag_filter:
            dialogWarning('Calculate filter parameters first!')
            return False

        try:
            f = float(self.editFrequency.text())
        except ValueError:
            dialogWarning('Frequency invalid!')
            return False

        if f <= 0:
            dialogWarning('Frequency must be positive!')
            return False
        if f > self._freqSampling/2:
            dialogWarning('Frequency above Nyquist frequency!')
            return False

        # calculate transient response
        T = 10/f
        N = int(T*self._freqSampling)
        ts = np.linspace(0, T, N)
        signal = np.sin(2*np.pi*f*ts)

        filt = IIRFilter(self._ff_coefs, self._fb_coefs)
        ys_filtered = []
        for i in range(N):
            ys_filtered.append(filt.update(signal[i]))

        # Plot
        self._canvas.prepare_axes()
        self._canvas.set_style()
        self._canvas.plot(
            ts,
            signal,
            label='input'
        )
        self._canvas.plot(
            ts,
            ys_filtered,
            label='filtered'
        )
        self._canvas.add_legend()
        self._canvas.refresh()

        return True


class FilterResponse(QTabWidget):
    
    def __init__(self):

        super().__init__()

        # tabs
        self._tabTransient = tabTransient()
        self._tabFrequency = tabFrequency()

        # layout
        self.addTab(self._tabTransient, "Transient")
        self.addTab(self._tabFrequency, "Frequency")

    def getCoefs(self, fb_coefs, ff_coefs, freqSampling):

        self._tabTransient.getCoefs(fb_coefs, ff_coefs, freqSampling)
        self._tabFrequency.getCoefs(fb_coefs, ff_coefs, freqSampling)