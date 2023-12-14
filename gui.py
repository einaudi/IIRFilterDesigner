# -*- coding: utf-8 -*-

import sys
import os

import yaml

import numpy as np

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


available_files = '(*.csv *.txt)'
tau_ext_margin = 0.1 # Hz


class FrequencyStability(QMainWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Configuration files
        widgets_conf = self.getWidgetsConfig()
        layout_conf = self.getLayoutConfig()

        # Variables
        self._data = None
        self._meta = None
        self._params = {}

        self._taus = [] # averaging times
        self._devs = {} # deviations
        self._conf_int = {} # confidence intervals
        self._noise_type = []

        self.initWidgets(widgets_conf)
        self.initLayout(layout_conf)
        self.initUI()

        print('Running application')
        self.show()

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

        print('Widgets initialised!')

    def initLayout(self, layout_conf):

        print('Initialising layout...')

        mainLayout = generate_layout(layout_conf, self._widgets)

        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)

        print('Layout initialised!')

    def initUI(self):

        pass

    def getParams(self):

        tmp = {}

        # Data length
        if self._data is None:
            dialogWarning('Load data first!')
            return False
        try:
            tmp['N'] = self._data['Frequency [Hz]'].size
        except KeyError:
            dialogWarning('Could not find frequency column in data!')
            return False
        
        # Parameters from settings section
        try:
            tmp['Central frequency [Hz]'] = float(self._widgets['freqCentral'].text())
            tmp['Sampling frequency [Hz]'] = float(self._widgets['freqSampling'].text())
            tmp['Tau min [s]'] = float(self._widgets['tauMin'].text())
            tmp['Tau max [s]'] = float(self._widgets['tauMax'].text())
            tmp['Tau N'] = int(self._widgets['tauN'].text())
        except ValueError:
            dialogWarning('Could not read parameters!')
            return False

        # Tau ranges
        if tmp['Tau min [s]'] < 1/tmp['Sampling frequency [Hz]']:
            dialogWarning('Minimal tau below sampling limit!')
            return False
        if tmp['Tau max [s]'] > tmp['N']/2/tmp['Sampling frequency [Hz]']:
            dialogWarning('Maximal tau above sampling limit!')
            return False
        if tmp['Tau max [s]'] <= tmp['Tau min [s]']:
            dialogWarning('Tau max lower or equal than tau min!')
            return False
        
        # Check if mean frequency option is enabled
        # Left for better resolution at small diferences between predicted and actual central frequency
        if self._widgets['checkCentral'].isChecked():
            tmp['Central frequency [Hz]'] = np.average(self._data['Frequency [Hz]'])
        
        self._params = tmp

        return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = FrequencyStability()
    sys.exit(app.exec_())
