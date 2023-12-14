import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import(
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.backends.backend_qt5agg import(
    NavigationToolbar2QT as NavigationToolbar)
import matplotlib.colors as colors

import numpy as np

from PyQt5.QtWidgets import QSizePolicy, QWidget, QVBoxLayout


class PlotCanvas(QWidget):

    def __init__(self, xLabel="", yLabel="", parent=None, toolbar=False):

        super().__init__()

        # Fig
        self._fig = Figure()
        # self._fig.patch.set_facecolor('None')
        self.canvas = FigureCanvas(self._fig)
        # self.canvas.setMinimumSize( 900, 550 )
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        # self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas.draw()

        self._xLabel = xLabel
        self.__yLabel = yLabel
        self._labels = None

        self._toolbar = toolbar

        # _axes
        self.init_axes()

        # Widgets
        if self._toolbar:
            self.toolbar = NavigationToolbar(self.canvas, parent)
        self.canvas.setParent(parent)

        self.init_layout()

    def init_axes(self):

        self._axes = self._fig.add_subplot(111)
        self._lns = list()

        self.axes = {
            'main' : self._axes
        }
        self.lns = {
            'main' : self._lns
        }

    def init_layout(self):

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        if self._toolbar:
            layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def set_style(self, axis='main', xLabel=None, yLabel=None):

        if xLabel is not None:
            self._xLabel = xLabel
        if yLabel is not None:
            self.__yLabel = yLabel

        self.axes[axis].set_xlabel(self._xLabel)
        self.axes[axis].set_ylabel(self.__yLabel)

    def prepare_axes(self, axis='main', **kwargs):
        """Clears axes and sets their settings

        Keyword arguments
        ------
        xLog : bool
        yLog : bool
        grid : bool
        """

        self.axes[axis].clear()
        self.lns[axis].clear()
        self.set_style()

        # print(kwargs)

        if kwargs.get('xLog'):
            self.axes[axis].set_xscale('log')
        if kwargs.get('yLog'):
            self.axes[axis].set_yscale('log')
        if kwargs.get('Grid'):
            self.axes[axis].grid(True)
        if 'x1Lim' in kwargs and 'x2Lim' in kwargs:
            self.axes[axis].set_xlim(
                kwargs['x1Lim'],
                kwargs['x2Lim']
            )
        if 'y1Lim' in kwargs and 'y2Lim' in kwargs:
            self.axes[axis].set_ylim(
                kwargs['y1Lim'],
                kwargs['y2Lim']
            )
        if kwargs.get('equal'):
            self.axes[axis].axis('equal')

    def disable_ticks(self, axis='main'):

        self.axes[axis].set_xticks([])
        self.axes[axis].set_xticklabels([])
        self.axes[axis].set_yticks([])
        self.axes[axis].set_yticklabels([])

    def plot(self, x, y, axis='main', to_legend=True, **kwargs):
        """Plot given data. Use xtickslabels if x is str.

        Parameters
        ------
        x : x axis data
        y : y axis data
        label : str

        Keyword arguments
        ------
        **kwargs for plt.plot
        """

        ln = self.axes[axis].plot(x, y, **kwargs)
        if to_legend:
            self._lns += ln

        return True

    def errorbar(self, x, y, axis='main', **kwargs):
        
        ln = self.axes[axis].errorbar(x, y, **kwargs)[0]
        if 'label' in kwargs.keys():
            ln.set_label(kwargs['label'])
        self._lns += [ln]

        return True

    def pcolormesh(self, xs, ys, zs, axis='main', **kwargs):

        xs_v, ys_v = np.meshgrid(xs, ys)

        self.axes[axis].pcolormesh(
            xs_v,
            ys_v,
            zs,
            cmap='inferno',
            shading='auto',
            **kwargs
        )
        
    def histogram(self, counts, bins, axis='main'):

        self.axes[axis].hist(
            bins[:-1],
            bins,
            weights=counts
        )

        return True

    def add_legend(self, axis='main', loc=0):

        labs = ['' if l.get_label().startswith('_') else l.get_label()
                for l in self.lns[axis]]

        try:
            self.axes[axis].legend(self.lns[axis], labs, loc=loc)
        except:
            pass

    def text(self, x, y, txt, axis='main', **kwargs):

        self.axes[axis].text(x, y, txt, **kwargs)

    def axvline(self, x, axis='main', **kwargs):

        self.axes[axis].axvline(x, **kwargs)

    def refresh(self):

        self._fig.tight_layout()
        self.canvas.draw()
        self.repaint()

    def get_ylim(self, axis='main'):

        ret = self.axes[axis].get_ylim()

        return ret
    
    def get_xlim(self, axis='main'):

        ret = self.axes[axis].get_xlim()

        return ret

    def set_ylim(self, y1, y2, axis='main'):

        self.axes[axis].set_ylim(y1, y2)

class PlotCanvasTwinx(PlotCanvas):

    def __init__(self, xLabel="", y1Label="", y2Label="", parent=None, **kwargs):

        super().__init__(
            xLabel=xLabel,
            yLabel=y1Label,
            parent=parent
        )

        self.__y2Label = y2Label
        
    def init_axes(self):

        self._axes = self._fig.add_subplot(111)
        self._lns = list()
        self._axesTwinx = self._axes.twinx()
        self._lnsTwinx = list()

        self.axes = {
            'main' : self._axes,
            'twinx' : self._axesTwinx
        }
        self.lns = {
            'main' : self._lns,
            'twinx' : self._lnsTwinx
        }

    def set_style_twinx(self, xLabel=None, y2Label=None):

        if xLabel is not None:
            self._xLabel = xLabel
        if y2Label is not None:
            self.__y2Label = y2Label

        self._axes.set_xlabel(self._xLabel)
        self._axesTwinx.set_ylabel(self.__y2Label)

    def prepare_axes_twinx(self, **kwargs):

        self._axesTwinx.clear()
        self._lnsTwinx.clear()
        self.set_style_twinx()
        # self._axesTwinx.callbacks.connect('xlim_changed', self.on_xlim_changed)

        if kwargs.get('xLog'):
            self._axesTwinx.set_xscale('log')
        if kwargs.get('yLog'):
            self._axesTwinx.set_yscale('log')
        if kwargs.get('grid'):
            self._axesTwinx.grid(True)

    # def add_legend(self, axis='main', loc=0):
            
    #     lns = self._lns + self._lnsTwinx
    #     labs = ['' if l.get_label().startswith('_') else l.get_label()
    #                     for l in lns]

    #     try:
    #         self._axes.legend(lns, labs, loc=loc)
    #     except:
    #         pass
