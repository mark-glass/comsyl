# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
__authors__ = ["M Glass - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"
__date__ = "20/04/2017"



import numpy as np
import scipy.signal as signal
import scipy.interpolate

class Convolution(object):
    def __init__(self):
        self._use_fft_convolve = True
        self._do_coordinate_check = True

    def _convolutionCoordinate(self, x):
        if x.shape[0] % 2 == 0:
            conv_x = np.linspace(x.min(), x.max(), x.shape[0]+1)
        else:
            conv_x = x

        return conv_x

    def _interpolateToOtherCoordinates(self, x, y, data, new_x, new_y):
        f_c = scipy.interpolate.RectBivariateSpline(x, y, data.real)
        result = f_c(new_x, new_y)

        f_c = scipy.interpolate.RectBivariateSpline(x, y, data.imag)
        result = result + 1j * f_c(new_x, new_y)

        return result

    def convolve1D(self, factor1, factor2):
        if self._use_fft_convolve:
            c=signal.fftconvolve(factor1, factor2, mode="same")
        else:
            c=signal.convolve(factor1, factor2, mode="same")
        return c

    def convolve2D(self, factor1, factor2):
        if self._use_fft_convolve:
            c=signal.fftconvolve(factor1, factor2, mode="same")
        else:
            c=signal.convolve(factor1, factor2, mode="same")
        return c

    def alignedConvolve2D(self, x, y, factor1, factor2):

        if self._do_coordinate_check:
            conv_x = self._convolutionCoordinate(x)
            conv_y = self._convolutionCoordinate(y)

            conv_factor1 = self._interpolateToOtherCoordinates(x, y, factor1, conv_x, conv_y)
            conv_factor2 = self._interpolateToOtherCoordinates(x, y, factor2, conv_x, conv_y)
        else:
            conv_factor1 = factor1
            conv_factor2 = factor2

        c = self.convolve2D(conv_factor1, conv_factor2)

        if self._do_coordinate_check:
            return self._interpolateToOtherCoordinates(conv_x, conv_y, c, x, y)
        else:
            return c

    def _seperateConvolve2D(self, factor1, factor2_x, factor2_y):
        # tmp = np.zeros_like(factor1)
        # result = np.zeros((factor1.shape[1], factor1.shape[0]), dtype=factor1.dtype)
        #
        #
        # # multiply should apply two every row
        # tmp = np.fft.fft(factor1, axis=1) * np.fft.fft(factor2_y)
        #
        # tmp = np.fft.ifft(tmp, axis=1)
        # # multiply should apply two every column but transposed so axis 1
        # tmp = np.fft.fft(tmp.transpose(), axis=1) * np.fft.fft(factor2_x)
        #
        # return np.fft.ifft(tmp, axis=1).transpose()

        tmp = np.zeros_like(factor1)
        result = np.zeros_like(factor1)

        for i_y in range(factor2_y.shape[0]):
            tmp[:, i_y] = signal.fftconvolve(factor1[:, i_y], factor2_x, mode="same")

        for i_x in range(factor2_x.shape[0]):
            result[i_x, :] = signal.fftconvolve(tmp[i_x,:], factor2_y, mode="same")

        return result