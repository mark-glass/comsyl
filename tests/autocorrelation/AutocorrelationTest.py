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



import unittest

import numpy as np

from comsyl.autocorrelation.SigmaMatrix import SigmaWaist
from comsyl.autocorrelation.AutocorrelationBuilder import AutocorrelationBuilder
from comsyl.autocorrelation.AutocorrelationBuilderStrategies import BuilderStrategyPython
from comsyl.autocorrelation.PhaseSpaceDensity import PhaseSpaceDensity
from comsyl.math.utils import plotSurface, createGaussian2D, norm2D


class ConvolutionTester(object):
    def __init__(self, Sigma_x, A, sigma_x):
        self._Sigma_x = Sigma_x
        self._A = A
        self._sigma_x = sigma_x

        max_x = max(Sigma_x, sigma_x) * 10
        self._x_coordinates = np.linspace(-max_x, max_x, 300)
        self._f = np.zeros(self._x_coordinates.shape[0])

    def coordinates(self):
        return self._x_coordinates

    def evaluateNumerical(self, x_1, x_2):
        for i_x, x in enumerate(self._x_coordinates):
            self._f[i_x] = np.exp(-x**2   /(2*self._Sigma_x**2)) * \
                           np.exp(-(x-x_1)**2/(2*self._sigma_x**2)) * \
                           np.exp(-(x-x_2)**2/(2*self._sigma_x**2))

        self._f *= self._A**2

        res = np.trapz(y=self._f, x=self._x_coordinates)

        return res

    def evaluateAnalytical(self, x_1, x_2):
        combined_sigma = self._Sigma_x**2 + 0.5 * self._sigma_x**2

        res =(self._A ** 2) * np.sqrt(2*np.pi)
        res *= np.sqrt(0.5/combined_sigma) * self._Sigma_x * self._sigma_x
        res *= np.exp( -(x_1-x_2)**2/(4*self._sigma_x**2) )
        res *= np.exp( -(x_1+x_2)**2/(8*combined_sigma) )

        return res


class EigenmoderTest(unittest.TestCase):

    @unittest.skip
    def testGaussian(self):
        x = np.linspace(-5,5,100)
        y = np.linspace(-5,5,100)

        f = createGaussian2D(1.0, 1.0, x, y)
        plotSurface(x, y, f, False)

    @unittest.skip
    def testConvolutionVisual(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                            sigma_y=1e-6,
                                            sigma_x_prime=5e-6,
                                            sigma_y_prime=5e-6)

        x_coordinates = np.linspace(-1e-5,1e-5,50)
        y_coordinates = np.linspace(-1e-5,1e-5,50)
        wavenumber = 1e+11

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        e_field = e_field + 0j

        af = AutocorrelationBuilder(N_e=0.0000001,
                             sigma_matrix=sigma_matrix,
                             field=e_field,
                             x_coordinates=x_coordinates,
                             y_coordinates=y_coordinates,
                             k=wavenumber)

        f = np.zeros((x_coordinates.shape[0],
                      y_coordinates.shape[0]), dtype=np.complex128)

        # damping along y should be higher than along x because sigma_x > sigma_y for the density.
        for y in (0.0 ,1.5e-6, -1.5e-6):
            for i_x_1, x_1 in enumerate(x_coordinates):
                for i_x_2, x_2 in enumerate(x_coordinates):
                    r_1 = np.array([x_1 + x_2, y])
                    r_2 = np.array([x_1 - x_2, y])

                    f[i_x_1, i_x_2] = af.evaluate(r_1, r_2)

            plotSurface(x_coordinates, x_coordinates, f, False)

        for x in (0.0, 1.5e-6, -1.5e-6):
            for i_y_1, y_1 in enumerate(y_coordinates):
                for i_y_2, y_2 in enumerate(y_coordinates):
                    r_1 = np.array([x, y_1 + y_2])
                    r_2 = np.array([x, y_1 - y_2])

                    f[i_y_1, i_y_2] = af.evaluate(r_1, r_2)

            plotSurface(y_coordinates, y_coordinates, f, False)

    @unittest.expectedFailure
    def testAnalyticalFormula(self):
        tester = ConvolutionTester(Sigma_x=1.0,
                                   A=2.0,
                                   sigma_x=1.0)

        for x_1 in tester.coordinates()[::5]:
            for x_2 in tester.coordinates()[::3]:
                diff = np.abs(tester.evaluateNumerical(x_1, x_2)-tester.evaluateAnalytical(x_1, x_2))
                self.assertLess(diff, 1e-10)

    def testConvolution(self):
        Sigma_x = 0.75e-6
        Sigma_y = 1e-6
        sigma_matrix = SigmaWaist(sigma_x=Sigma_x,
                                  sigma_y=Sigma_y,
                                  sigma_x_prime=1e-60,
                                  sigma_y_prime=1e-60,
                                  sigma_dd=0.89e-03)

        x_coordinates = np.linspace(-0.5e-5, 0.5e-5, 200)
        y_coordinates = np.linspace(-0.5e-5, 0.5e-5, 200)
        wavenumber = 1e+11

        sigma_x = 1.0e-6
        sigma_y = 1.0e-6
        e_field = createGaussian2D(sigma_x=sigma_x,
                                   sigma_y=sigma_y,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        e_field = e_field + 0j
        e_field = e_field[np.newaxis, :, :]

        tester_x = ConvolutionTester(Sigma_x=Sigma_x,
                                     A=2.0,
                                     sigma_x=sigma_x)
        tester_y = ConvolutionTester(Sigma_x=Sigma_y,
                                     A=2.0,
                                     sigma_x=sigma_y)


        af = AutocorrelationBuilder(N_e=0.0000001,
                                    sigma_matrix=sigma_matrix,
                                    weighted_fields=e_field,
                                    x_coordinates=x_coordinates,
                                    y_coordinates=y_coordinates,
                                    k=wavenumber)

        f = np.zeros((x_coordinates.shape[0],
                      y_coordinates.shape[0]), dtype=np.complex128)
        t = np.zeros_like(f)

        # Test along y slices
        for y in y_coordinates:
            for i_x_1, x_1 in enumerate(x_coordinates):
                for i_x_2, x_2 in enumerate(x_coordinates):
                    r_1 = np.array([x_1 + x_2, y])
                    r_2 = np.array([x_1 - x_2, y])

                    f[i_x_1, i_x_2] = af.evaluate(r_1, r_2)
                    t[i_x_1, i_x_2] = tester_x.evaluateAnalytical(r_1[0], r_2[0]) * tester_y.evaluateAnalytical(r_1[1], r_2[1])

            f /= norm2D(x_coordinates, x_coordinates, f)
            t /= norm2D(x_coordinates, x_coordinates, t)

            diff = norm2D(x_coordinates, x_coordinates, f-t)

            plotSurface(x_coordinates, x_coordinates, f)
            plotSurface(x_coordinates, x_coordinates, t)

            self.assertLess(diff, 1e-10)

        # Test along x slices
        for x in x_coordinates:
            for i_y_1, y_1 in enumerate(y_coordinates):
                for i_y_2, y_2 in enumerate(y_coordinates):
                    r_1 = np.array([x, y_1 + y_2])
                    r_2 = np.array([x, y_1 - y_2])

                    f[i_y_1, i_y_2] = af.evaluate(r_1, r_2)
                    t[i_y_1, i_y_2] = tester_x.evaluateAnalytical(r_1[0],r_2[0]) * tester_y.evaluateAnalytical(r_1[1], r_2[1])

            f /= norm2D(y_coordinates, y_coordinates, f)
            t /= norm2D(y_coordinates, y_coordinates, t)

            diff = norm2D(y_coordinates, y_coordinates, f-t)
            print(diff)
            self.assertLess(diff, 1e-10)

    @unittest.skip
    def testFredholm(self):
        Sigma_x = 0.75e-6
        Sigma_y = 1e-6
        sigma_matrix = SigmaWaist(sigma_x=Sigma_x,
                                            sigma_y=Sigma_y,
                                            sigma_x_prime=1e-6,
                                            sigma_y_prime=1e-6)

        x_coordinates = np.linspace(-1e-5, 1e-5, 60)
        y_coordinates = np.linspace(-1e-5, 1e-5, 60)
        wavenumber = 1e+11

        sigma_x=1.0e-6
        sigma_y=1.0e-6
        e_field = createGaussian2D(sigma_x=sigma_x,
                                   sigma_y=sigma_y,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        e_field = e_field + 0j

        af = AutocorrelationBuilder(N_e=0.0000001,
                             sigma_matrix=sigma_matrix,
                             field=e_field,
                             x_coordinates=x_coordinates,
                             y_coordinates=y_coordinates,
                             k=wavenumber)

        x_coordinates = af._field_x_coordinates
        y_coordinates = af._field_y_coordinates

        f2d = np.zeros((x_coordinates.shape[0],
                        y_coordinates.shape[0]), dtype=np.complex128)
        f = np.zeros((x_coordinates.shape[0]* y_coordinates.shape[0]), dtype=np.complex128)

        r_1 = np.array([x_coordinates[0], y_coordinates[0]])
        r_2 = np.array([x_coordinates[0], y_coordinates[0]])
        res = af.evaluate(r_1, r_2)

        f2d[0,0] = 1.0
        res_f = af.fredholmAction(f2d)

        print(res)
        print(res_f[0,0])

    def testCalculateRhoPhase(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                            sigma_y=1e-6,
                                            sigma_x_prime=5e-6,
                                            sigma_y_prime=5e-6)

        x_coordinates = np.linspace(-1e-5,1e-5,50)
        y_coordinates = np.linspace(-1e-5,1e-5,50)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        e_field = e_field + 0j

        strategy = BuilderStrategyPython(x_coordinates,y_coordinates,density, x_coordinates,y_coordinates, e_field[np.newaxis, :,:])

        for x in x_coordinates[::5]:
            for y in y_coordinates[::2]:
                norm_diff = np.linalg.norm(strategy.calculateRhoPhase((x, y)) - strategy.calculateRhoPhaseSlow((x, y)))
                self.assertLess(norm_diff, 1e-10)