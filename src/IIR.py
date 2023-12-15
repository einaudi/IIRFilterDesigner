# -*- coding: utf-8 -*-

import numpy as np


class IIRFilter():

    def __init__(self, ff_coefs, fb_coefs):

        # multiplies inputs
        self._ff_coefs = np.array(ff_coefs)
        self._ff_order = self._ff_coefs.size
        # print('Filter feedforward order: ', self._ff_order)

        # multiplies outputs
        self._fb_coefs = np.array(fb_coefs)
        self._fb_order = self._fb_coefs.size
        # print('Filter feedback order: ', self._fb_order)

        self._input = np.zeros(self._ff_order)
        self._output = np.zeros(self._fb_order)

    def update(self, x):

        self._input[1:] = self._input[0:-1]
        self._input[0] = x

        y = np.sum(self._ff_coefs * self._input)
        y += np.sum(self._fb_coefs * self._output)

        self._output[1:] = self._output[0:-1]
        self._output[0] = y

        return y

    def reset(self):

        self._input = np.zeros(self._ff_order)
        self._output = np.zeros(self._fb_order)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    a = 0.8
    N = 50

    xs = np.ones(N)
    xs[:5] = 0

    ys = np.zeros(N)

    filt = IIRFilter(1-a, a)

    for i in range(N):
        ys[i] = filt.update(xs[i])

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    ax.plot(range(N), xs, label='setpoint')
    ax.plot(range(N), ys, label='filter')

    ax.legend(loc=0)

    fig.savefig('./test_filter.png')