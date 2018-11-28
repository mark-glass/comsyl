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

from comsyl.mathcomsyl.utils import Enum

class Unitary(Enum):
    def __init__(self):
        Enum.__init__(self, "unitary")

class Backward(Enum):
    def __init__(self):
        Enum.__init__(self, "backward")

class FourierTransform(object):
    def __init__(self, scaling=1.0, constant_fashion=Unitary()):
        self._scaling = scaling
        self._constant_fashion = constant_fashion

    def forwardFT(self, x_coordinates, f_values):
        delta_coordinates = x_coordinates[1] - x_coordinates[0]

        f_hat = np.fft.fft(f_values)
        k_coordinates = np.fft.fftfreq(f_values.size)* 2 * np.pi / delta_coordinates

        exp_argument = -1j*k_coordinates*x_coordinates[0]
        f_hat *= (1.0 / np.sqrt(2*np.pi)) * \
                 delta_coordinates * \
                 np.exp(exp_argument)

        if self._constant_fashion==Unitary():
            f_hat *= self._scaling
        elif self._constant_fashion==Backward():
            f_hat *= np.sqrt(2*np.pi)
        else:
            raise NotImplementedError

        return k_coordinates, f_hat

    def backwardFT(self, k_coordinates, f_values):
        delta_coordinates = k_coordinates[1] - k_coordinates[0]

        f = np.fft.ifft(f_values)
        x_coordinates = (np.fft.fftfreq(f_values.size)* 2 * np.pi) / delta_coordinates

        exp_argument = 1j*x_coordinates*k_coordinates[0]
        f *= (1.0 / (np.sqrt(2*np.pi))) * \
              delta_coordinates * \
              np.exp(exp_argument)

        f *= f_values.size

        if self._constant_fashion==Unitary():
            f *= self._scaling
        elif self._constant_fashion==Backward():
            # One sqrt is already multiplied.
            f *= (1.0/np.sqrt(2*np.pi)) * self._scaling**2
        else:
            raise NotImplementedError

        return x_coordinates, f

    def testedForwardFT(self, x_coordinates, f_values):
        k_coordinates, f_hat = self.forwardFT(x_coordinates, f_values)

        x_coordinates_test, f_test_values = self.backwardFT(k_coordinates, f_hat)

        sort_indices = np.argsort(x_coordinates_test)
        from scipy.interpolate import interp1d
        f_test = interp1d(x_coordinates_test[sort_indices], f_test_values[sort_indices])

        accuracy = 0.0

        i_elements = 0
        min_x_test = x_coordinates_test.min()
        max_x_test = x_coordinates_test.max()
        for i_x, x in enumerate(x_coordinates):
            if min_x_test <= x <= max_x_test:
                accuracy += np.abs(self._scaling**2 * f_values[i_x]-f_test(x))**2
                i_elements += 1

        accuracy = np.sqrt(accuracy) * (x_coordinates.max() - x_coordinates.min()) / i_elements


        return k_coordinates, f_hat, accuracy

    def testedBackwardFT(self, k_coordinates, f_values):
        x_coordinates, f = self.backwardFT(k_coordinates, f_values)

        k_coordinates_test, f_hat_test_values = self.forwardFT(x_coordinates, f)

        sort_indices = np.argsort(k_coordinates_test)
        from scipy.interpolate import interp1d
        f_test = interp1d(k_coordinates_test[sort_indices], f_hat_test_values[sort_indices])

        accuracy = 0.0

        i_elements = 0
        min_k_test = k_coordinates_test.min()
        max_k_test = k_coordinates_test.max()
        for i_k, k in enumerate(k_coordinates):
            if min_k_test <= k <= max_k_test:
                accuracy += np.abs(self._scaling**2 * f_values[i_k]-f_test(k))**2
                i_elements += 1

        accuracy = np.sqrt(accuracy) * (k_coordinates.max() - k_coordinates.min())/ i_elements


        return x_coordinates, f, accuracy

    def forwardFT2D(self, x_coordinates, y_coordinates, values):
        f_hat = np.zeros((len(x_coordinates),len(y_coordinates)), dtype=np.complex128)

        for i_y in range(len(y_coordinates)):
            k_x, f_hat_x = self.forwardFT(x_coordinates, values[:, i_y])
            f_hat[:,i_y] = f_hat_x

        for i_x in range(len(x_coordinates)):
            k_y, f_hat_y = self.forwardFT(y_coordinates, f_hat[i_x, :])
            f_hat[i_x, :] = f_hat_y

        return k_x, k_y, f_hat

    def forwardFT2DInt(self, x_coordinates, y_coordinates, values, k_x, k_y):
        f_e_x = np.exp(-1j*k_x*x_coordinates)
        f_e_y = np.exp(-1j*k_y*y_coordinates)

        f_e = np.outer(f_e_x,f_e_y)
        integrand = f_e * values
        from comsyl.math.utils import trapez2D
        f_k_xy = trapez2D(integrand,1,1)

        return f_k_xy


    def backwardFT2D(self, k_x_coordinates, k_y_coordinates, values):
        f= np.zeros((len(k_x_coordinates),len(k_y_coordinates)), dtype=np.complex128)

        for i_y in range(len(k_y_coordinates)):
            x, f_x = self.backwardFT(k_x_coordinates, values[:, i_y])
            f[:,i_y] = f_x

        for i_x in range(len(k_x_coordinates)):
            y, f_y = self.backwardFT(k_y_coordinates, f[i_x, :])
            f[i_x, :] = f_y

        return x, y, f
