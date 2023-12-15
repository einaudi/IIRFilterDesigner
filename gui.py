# -*- coding: utf-8 -*-

import sys
import os
import time
import threading

import yaml

import numpy as np

import src.filter_calc as fc
from src.IIR import IIRFilter
from misc.generators import generate_widgets, generate_layout
from widgets.Dialogs import *

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox
)
from PyQt5.QtCore import pyqtSignal


available_files = '(*.csv *.txt)'
tau_ext_margin = 0.1 # Hz


class IIRFilterCalculator(QMainWindow):

    updateProgress = pyqtSignal(int)
    updateETA = pyqtSignal(float, float)
    plotTransfer = pyqtSignal()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Configuration files
        widgets_conf = self.getWidgetsConfig()
        layout_conf = self.getLayoutConfig()

        # Variables
        self._params = {}
        self._digitalZeros = []
        self._digitalPoles = []
        self._fb_coefs = []
        self._ff_coefs = []

        self._flag_digital_calculated = False

        self._eventStop = None
        self._threadTest = None

        # Transfer functions
        self._H = lambda s: 1
        self._H_z = lambda z: 1
        self._fs_transfer = np.zeros(1)
        self._ys_analog = np.zeros(1)
        self._ys_analog_phase = np.zeros(1)
        self._ys_digital = np.zeros(1)
        self._ys_digital_phase = np.zeros(1)
        self._ys_implemented = np.zeros(1)

        self.initWidgets(widgets_conf)
        self.initLayout(layout_conf)
        self.initUI()

        print('Running application')
        self.show()

    def __exit__(self):

        try:
            self._testFilterCancel
        except:
            pass

    def getWidgetsConfig(self):

        config_path = os.path.join("./", "config", "widgets_config.yml")
        with open(config_path) as config_file:
            widgets_conf = yaml.safe_load(config_file)

        return widgets_conf

    def getLayoutConfig(self):

        config_path = os.path.join("./", "config", "layout_config.yml")
        with open(config_path) as config_file:
            layout_conf = yaml.safe_load(config_file)

        return layout_conf

    def initWidgets(self, widget_conf):

        print('Initialising widgets...')

        self._widgets = generate_widgets(widget_conf)

        # Fine settings of poles, zeros canvas
        self._widgets['canvasPolesZeros']._fig.set_figwidth(2)
        self._widgets['canvasPolesZeros']._fig.set_figheight(2)
        self._widgets['canvasPolesZeros'].axes['main'].axis('off')
        self._plot_poles_zeros()

        print('Widgets initialised!')

    def initLayout(self, layout_conf):

        print('Initialising layout...')

        mainLayout = generate_layout(layout_conf, self._widgets)

        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)

        print('Layout initialised!')

    def initUI(self):

        self._widgets['btnCalcAnalog'].clicked.connect(lambda : self._calcAnalogFilter())
        self._widgets['btnCalcDigital'].clicked.connect(self._calcDigitalFilter)
        self._widgets['btnTestFilter'].clicked.connect(self._testFilter)
        self._widgets['btnTestCancel'].clicked.connect(self._testFilterCancel)

        self.updateProgress.connect(lambda cts: self._widgets['progressBar'].setValue(cts))
        self.updateETA.connect(lambda time_left, speed: self._widgets['labelProgress'].setText('eta: {0:6.2f} s ({1:6.2f}/s)'.format(
            time_left,
            speed
        )))
        self.plotTransfer.connect(self._plot_transfers)

    def getParams(self):

        tmp = {}
        
        # Parameters from settings section
        try:
            tmp['Passband frequency [Hz]'] = float(self._widgets['freqPassband'].text())
            tmp['Stopband frequency [Hz]'] = float(self._widgets['freqStopband'].text())
            tmp['Passband attenuation [dB]'] = float(self._widgets['attPassband'].text())
            tmp['Stopband attenuation [dB]'] = float(self._widgets['attStopband'].text())
            tmp['Sampling frequency [Hz]'] = float(self._widgets['freqSampling'].text())
            tmp['Filter gain'] = float(self._widgets['filterGain'].text())
        except ValueError:
            dialogWarning('Could not read parameters!')
            return False

        # Ranges
        if tmp['Passband frequency [Hz]'] > tmp['Sampling frequency [Hz]']/2:
            dialogWarning('Passband frequency above Nyquist frequency!')
            return False
        if tmp['Stopband frequency [Hz]'] > tmp['Sampling frequency [Hz]']/2:
            dialogWarning('Stopband frequency above Nyquist frequency!')
            return False
        if tmp['Passband frequency [Hz]'] > tmp['Stopband frequency [Hz]']:
            dialogWarning('Passband frequency above stopband frequency!')
            return False
        if tmp['Filter gain'] < 1:
            dialogWarning('Filter gain below unity')
            return False
        
        self._params = tmp

        self._fs_transfer = np.logspace(
            np.log10(self._params['Passband frequency [Hz]']/100),
            np.log10(self._params['Sampling frequency [Hz]']/2),
            100
        )

        return True

    # Calculations
    def _calcAnalogFilter(self, plot=True):

        if not self.getParams():
            return False

        # Calculate Nyquist frequency
        self._params['Nyquist frequency [Hz]'] = self._params['Sampling frequency [Hz]']/2
        self._widgets['freqNyquist'].setText('{:.4e}'.format(self._params['Nyquist frequency [Hz]']))

        # Translate frequencies to angular
        self._params['Passband Omega'] = 2*np.pi*self._params['Passband frequency [Hz]']
        self._params['Stopband Omega'] = 2*np.pi*self._params['Stopband frequency [Hz]']
        self._params['Sampling Omega'] = 2*np.pi*self._params['Sampling frequency [Hz]']
        self._params['Nyquist Omega'] = 2*np.pi*self._params['Nyquist frequency [Hz]']

        # Translate attenuation to relative
        gain_offset_dB = fc.att_to_dB(self._params['Filter gain'])
        self._params['Passband attenuation'] = fc.dB_to_att(self._params['Passband attenuation [dB]'] - gain_offset_dB)
        self._params['Stopband attenuation'] = fc.dB_to_att(self._params['Stopband attenuation [dB]'] - gain_offset_dB)

        # Calculate filter order
        self._params['Filter order'] = fc.calc_order(
            self._params['Passband Omega'],
            self._params['Stopband Omega'],
            self._params['Passband attenuation'],
            self._params['Stopband attenuation']
        )
        self._widgets['filterOrder'].setText('{}'.format(self._params['Filter order']))

        # Calculate cutoff frequency
        self._params['Cutoff Omega'] = fc.calc_cutoff_freq(
            self._params['Stopband Omega'],
            self._params['Stopband attenuation'],
            self._params['Filter order']
        )
        self._params['Cutoff frequency [Hz]'] = self._params['Cutoff Omega']/2/np.pi
        self._widgets['freqCutoff'].setText('{:.4e}'.format(self._params['Cutoff frequency [Hz]']))

        # Calculate filter roots
        self._params['Filter roots'] = fc.calc_normalised_lowpass_roots(self._params['Filter order'])

        # Calculate analog transfer function
        H = fc.get_continuous_transfer_function(
            self._params['Cutoff Omega'],
            self._params['Filter order'],
            self._params['Filter roots']
        )
        self._H = lambda s: self._params['Filter gain'] * H(s)

        if plot:
            self._plot_analog()

        return True

    def _calcDigitalFilter(self):

        if not self._calcAnalogFilter(plot=False):
            return False

        H_z = fc.get_digital_transfer_function(
            self._params['Cutoff Omega'],
            self._params['Filter order'],
            self._params['Filter roots'],
            self._params['Sampling frequency [Hz]']
        )
        self._H_z = lambda z: self._params['Filter gain'] * H_z(z)

        z, p, k = fc.get_digital_filter_zpk(
            self._params['Filter order'],
            self._params['Cutoff Omega'],
            self._params['Sampling Omega']
        )
        self._digitalZeros = z
        self._digitalPoles = p

        fb_coefs, ff_coefs, _ = fc.get_digital_filter_coefs(
            self._params['Filter order'],
            self._params['Cutoff Omega'],
            self._params['Sampling Omega']
        )
        self._fb_coefs = fb_coefs
        self._ff_coefs = self._params['Filter gain'] * ff_coefs

        self._plot_analog(False)
        self._plot_digital()
        self._plot_poles_zeros()

        self._showResults()

        self._flag_digital_calculated = True
        return True

    def _showResults(self):

        msg = 'Calculated filter parameters:\n'
        # filter coefs
        msg += 'Feedback coefs: ['
        for item in self._fb_coefs:
            msg += '{:.4e}, '.format(item)
        msg = msg[:-2]
        msg += ']\n'
        msg += 'Feedforward coefs: ['
        for item in self._ff_coefs:
            msg += '{:.4e}, '.format(item)
        msg = msg[:-2]
        msg += ']\n'
        # Zeros and poles
        msg += 'Zeros: ['
        for item in self._digitalZeros:
            msg += '{:.4e}, '.format(item)
        msg = msg[:-2]
        msg += ']\n'
        msg += 'Poles: ['
        for item in self._digitalPoles:
            msg += '{:.4e}, '.format(item)
        msg = msg[:-2]
        msg += ']\n'

        self._widgets['calcResults'].setText(msg)

    # Filter testing
    def _testFilter(self):

        if not self._flag_digital_calculated:
            dialogWarning('Calculate filter parameters first!')
            return False

        print('Starting test filter thread...')
        self._eventStop = threading.Event()
        self._threadTest = threading.Thread(target=self._testFilterLoop, args=(self._eventStop,))

        self._threadTest.start()

    def _testFilterLoop(self, eventStop):
        
        self._widgets['btnCalcAnalog'].setEnabled(False)
        self._widgets['btnCalcDigital'].setEnabled(False)
        self._widgets['btnTestFilter'].setEnabled(False)

        self._widgets['progressBar'].setMaximum(self._fs_transfer.size)
        ys_amp = []
        filt = IIRFilter(
            self._ff_coefs,
            self._fb_coefs,
        )
        start = time.time()
        for i, f in enumerate(self._fs_transfer):
            self.updateProgress.emit(i)
            stop = time.time()
            speed = 1/(stop-start)
            start = time.time()
            try:
                time_left = (self._fs_transfer.size - i)/speed
            except ZeroDivisionError:
                time_left = -1
            self.updateETA.emit(time_left, speed)

            T = 50/f # check filter on 200 periods
            N = int(T*self._params['Sampling frequency [Hz]']) # set time step according to sampling frequency
            ts = np.linspace(0, T, N)
            signal = 0.5*np.sin(2*np.pi*f*ts)
            tmp = []
            filt.reset()
            for i in range(N):
                tmp.append(filt.update(signal[i]))
            
            n = 5*int(self._params['Sampling frequency [Hz]']/f) # number of points for 5 periods
            ys_amp.append(np.amax(tmp[-n:]) - np.amin(tmp[-n:]))

            if eventStop.is_set():
                return False

        self._ys_implemented = fc.att_to_dB(ys_amp)

        self.plotTransfer.emit()

        self._widgets['btnCalcAnalog'].setEnabled(True)
        self._widgets['btnCalcDigital'].setEnabled(True)
        self._widgets['btnTestFilter'].setEnabled(True)

    def _testFilterCancel(self):

        if self._eventStop is not None:
            self._eventStop.set()
            self._threadTest.join()
        self._widgets['btnCalcAnalog'].setEnabled(True)
        self._widgets['btnCalcDigital'].setEnabled(True)
        self._widgets['btnTestFilter'].setEnabled(True)

    # Plotting
    def _plot_analog(self, refresh=True):

        # Calculate filter transfer
        tmp = self._H(1j*2*np.pi*self._fs_transfer)
        self._ys_analog = fc.att_to_dB(np.absolute(tmp))
        self._ys_analog_phase = np.angle(tmp) / np.pi * 180

        # Plot analog transfer function
        self._widgets['canvasTransfer'].prepare_axes(xLog=True, Grid=True)
        self._widgets['canvasTransfer'].prepare_axes_twinx(xLog=True, Grid=True)
        self._widgets['canvasTransfer'].set_style()
        self._widgets['canvasTransfer'].set_style_twinx()

        self._widgets['canvasTransfer'].plot(
            self._fs_transfer,
            self._ys_analog,
            color='C0',
            label='continuous'
        )
        self._widgets['canvasTransfer'].plot(
            self._fs_transfer,
            self._ys_analog_phase,
            axis='twinx',
            color='C0',
            linestyle='dashed',
            to_legend=False
        )

        # Plot critical parameters
        self._widgets['canvasTransfer'].plot(
            [self._params['Passband frequency [Hz]'], self._params['Stopband frequency [Hz]']],
            [self._params['Passband attenuation [dB]'], self._params['Stopband attenuation [dB]']],
            marker='o',
            linewidth=0,
            markersize=4,
            color='r',
            zorder=10,
            to_legend=False
        )

        self._widgets['canvasTransfer'].add_legend()
        if refresh:
            self._widgets['canvasTransfer'].refresh()

    def _plot_digital(self, refresh=True):

        # Warp frequency for calculations
        omegas = fc.warp_frequency(2*np.pi*self._fs_transfer, self._params['Sampling frequency [Hz]'])

        # Calculate transfer
        tmp = self._H_z(np.exp(1j*omegas))
        self._ys_digital = fc.att_to_dB(np.absolute(tmp))
        self._ys_digital_phase = np.angle(tmp) / np.pi*180

        # Plot transfer
        self._widgets['canvasTransfer'].plot(
            self._fs_transfer,
            self._ys_digital,
            color='C1',
            label='discrete'
        )
        self._widgets['canvasTransfer'].plot(
            self._fs_transfer,
            self._ys_digital_phase,
            axis='twinx',
            color='C1',
            linestyle='dashed',
            to_legend=False
        )

        self._widgets['canvasTransfer'].add_legend()
        if refresh:
            self._widgets['canvasTransfer'].refresh()

    def _plot_transfers(self):
        
        self._plot_analog(False)
        self._plot_digital(False)
        self._widgets['canvasTransfer'].plot(
            self._fs_transfer,
            self._ys_implemented,
            color='C2',
            label='implemented'
        )
        self._widgets['canvasTransfer'].add_legend()
        self._widgets['canvasTransfer'].refresh()

    def _plot_poles_zeros(self):

        self._widgets['canvasPolesZeros'].prepare_axes()
        self._widgets['canvasPolesZeros'].disable_ticks()
        self._widgets['canvasPolesZeros'].axes['main'].axis('off')

        thetas = np.linspace(0, 2*np.pi, 100)
        xs_circle = np.cos(thetas)
        ys_circle = np.sin(thetas)

        self._widgets['canvasPolesZeros'].plot(
            xs_circle,
            ys_circle,
            linestyle='dashed',
            linewidth=0.5,
            color='k'
        )

        # Plot zeros and poles
        xs_z = [item.real for item in self._digitalZeros]
        ys_z = [item.imag for item in self._digitalZeros]
        xs_p = [item.real for item in self._digitalPoles]
        ys_p = [item.imag for item in self._digitalPoles]

        self._widgets['canvasPolesZeros'].plot(
            xs_z,
            ys_z,
            linewidth=0,
            marker='o',
            markersize=4
        )
        self._widgets['canvasPolesZeros'].plot(
            xs_p,
            ys_p,
            linewidth=0,
            marker='x',
            markersize=4
        )

        self._widgets['canvasPolesZeros'].refresh()
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = IIRFilterCalculator()
    sys.exit(app.exec_())
