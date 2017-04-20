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
import scipy.misc as misc
import scipy.special as special

class GaussianSchellModel1D(object):
    def __init__(self, A, sigma_s, sigma_g):
        """
        Mandel and Wolf p 253

        :param A: Amplitude of the spectral density S
        :param sigma_s: Gaussian width of the spectral density S
        :param sigma_g: Gaussian width of spectral degree of coherence g
        """
        self._A = A
        self._sigma_s = sigma_s
        self._sigma_g = sigma_g

    def S(self, x):
        S = (self._A ** 2) * np.exp(-(x**2)/(2*self._sigma_s**2))
        return S

    def g(self, x):
        g = np.exp(-(x**2)/(2*self._sigma_g**2))
        return g

    def evaluate(self, x_1, x_2):

        dr = x_1-x_2

        S_1 = self.S(x_1)
        S_2 = self.S(x_2)
        g = self.g(dr)

        result = np.sqrt(S_1) * np.sqrt(S_2) * g

        return result

    def a(self):
        a = 1.0/(4.0*self._sigma_s**2)
        return a

    def b(self):
        b = 1.0/(2.0*self._sigma_g**2)
        return b

    def c(self):
        a = self.a()
        b = self.b()

        res = np.sqrt(a**2 + 2 * a * b)
        return res

    def beta(self,n):
        a = self.a()
        b = self.b()
        c = self.c()

        res = self._A**2 * np.sqrt(np.pi/(a+b+c)) * (b/(a+b+c))**n
        return res

    def phi(self, n, x):
        c_h_n = special.hermite(n)
        c = self.c()

        #res = np.sqrt(2.0*c)**0.5 * (1/np.pi)**0.25 * (1.0/np.sqrt(2**n * misc.factorial(n) ))
        #res =  (1.0/np.sqrt(np.sqrt(np.pi) * 2**n * misc.factorial(n) ))**0.5
        #h_n = np.polyval(c_h_n, x * np.sqrt(2*c))
        h_n = np.polyval(c_h_n, x * np.sqrt(2*c) )

#        h_n *= 2 ** -(n/2.0)

        res = ((2*c/np.pi) ** 0.25) * np.exp(-c * x**2)
        res *= h_n * (1.0/np.sqrt(2**n * misc.factorial(n) ))

        return res

class GaussianSchellModel2D(object):
    def __init__(self, A, sigma_s_x, sigma_g_x, sigma_s_y, sigma_g_y):
        """
        Mandel and Wolf p 253

        :param A: Amplitude of the spectral density S
        :param sigma_s: Gaussian width of the spectral density S
        :param sigma_g: Gaussian width of spectral degree of coherence g
        """

        self._mode_x = GaussianSchellModel1D(A**0.5, sigma_s_x, sigma_g_x)
        self._mode_y = GaussianSchellModel1D(A**0.5, sigma_s_y, sigma_g_y)

        # For eigenvalue ordering.
        self._sorted_mode_indices = None

    def evaluate(self, r_1, r_2):
        x = self._mode_x.evaluate(r_1[0], r_2[0])
        y = self._mode_y.evaluate(r_1[1], r_2[1])

        result = x * y

        return result

    def beta(self, n_x, n_y):
        beta_x = self._mode_x.beta(n_x)
        beta_y = self._mode_y.beta(n_y)

        res = beta_x * beta_y

        return res

    def phi(self, n_x, n_y, x, y):
        phi_x = self._mode_x.phi(n_x, x)
        phi_y = self._mode_y.phi(n_y, y)

        res = phi_x * phi_y

        #print(x, y, phi_x, phi_y, res)
        return res

    def phi_nm(self,n_x, n_y, x_coords, y_coords):
        phi_x = self._mode_x.phi(n_x, x_coords)
        phi_y = self._mode_y.phi(n_y, y_coords)

        res = np.outer(phi_x,  phi_y)
        return res

    def sortedModeIndices(self, index_energy, n_points=50):
        if self._sorted_mode_indices is None:
            eigenvalues_x = np.array([self._mode_x.beta(i) for i in range(n_points)])
            eigenvalues_y = np.array([self._mode_y.beta(i) for i in range(n_points)])

            f = np.outer(eigenvalues_x, eigenvalues_y)
            self._sorted_mode_indices = np.array(np.unravel_index(f.flatten().argsort()[::-1], (n_points, n_points)))

        n, m = self._sorted_mode_indices[:, index_energy]
        return n, m
