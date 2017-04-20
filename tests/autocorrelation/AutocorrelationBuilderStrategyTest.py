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
import unittest

from scipy.ndimage.interpolation import shift

from comsyl.autocorrelation.AutocorrelationBuilderStrategies import BuilderStrategyPython, BuilderStrategyConvolution
from comsyl.autocorrelation.PhaseSpaceDensity import PhaseSpaceDensity
from comsyl.autocorrelation.SigmaMatrix import SigmaWaist, SigmaMatrixFromCovariance
from tests.parallel.ParallelVectorTest import createVector
from comsyl.math.utils import plotSurface, createGaussian2D, norm2D

def calculateRhoPhaseSlow(strategy, dr):
    i_x = strategy.xIndexByCoordinate(-dr[0])
    i_y = strategy.yIndexByCoordinate(-dr[1])

    for i_r_x, r_x in enumerate(strategy._field_x_coordinates):
        for i_r_y, r_y in enumerate(strategy._field_y_coordinates):
            a, b = (strategy._rho_phase_x[i_x] *r_x, strategy._rho_phase_y[i_y]* r_y)
            strategy._rho_phase_tmp[i_r_x, i_r_y] = np.exp(a+b)

    return strategy._rho_phase_tmp


class AutocorrelationBuilderStrategyTest(unittest.TestCase):

    def testCalculateRhoPhaseNoAlpha(self):
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

        strategy = BuilderStrategyPython(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])
        strategy._setUpPhases()

        for x in x_coordinates[::5]:
            for y in y_coordinates[::4]:
                norm_diff = np.linalg.norm(strategy.calculateRhoPhase((x, y)) - calculateRhoPhaseSlow(strategy, (x, y)))
                self.assertLess(norm_diff, 1e-10)

    def testCalculateRhoPhase(self):
        sigma_matrix = SigmaMatrixFromCovariance(xx=3e-6,
                                            yy=1e-6,
                                            xpxp=5e-6,
                                            ypyp=5e-6,
                                            xxp=1e-6,
                                            yyp=5e-7)

        x_coordinates = np.linspace(-1e-6,1e-6, 51)
        y_coordinates = np.linspace(-1e-6,1e-6, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        e_field = e_field + 0j

        strategy = BuilderStrategyPython(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])

        for x in x_coordinates[::5]:
            for y in y_coordinates[::2]:
                a = strategy.calculateRhoPhase((x, y)).copy()
                b = calculateRhoPhaseSlow(strategy, (x, y)).copy()
                norm_diff = np.linalg.norm(a - b)
                self.assertLess(norm_diff, 1e-13)

    def testCalculateRhoPhaseTrace(self):
        sigma_matrix = SigmaMatrixFromCovariance(xx=3e-6,
                                            yy=1e-6,
                                            xpxp=5e-6,
                                            ypyp=5e-6,
                                            xxp=1e-6,
                                            yyp=5e-7)

        x_coordinates = np.linspace(-1e-5, 1e-5, 51)
        y_coordinates = np.linspace(-1e-5, 1e-5, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        e_field = e_field + 0j

        strategy = BuilderStrategyPython(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])

        self.assertLess(np.linalg.norm(strategy.calculateRhoPhase((0, 0))-1), 1e-12)

    def testRelativeShiftedCopy(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                            sigma_y=1e-6,
                                            sigma_x_prime=5e-6,
                                            sigma_y_prime=5e-6)

        x_coordinates = np.linspace(-1e-5, 1e-5, 51)
        y_coordinates = np.linspace(-1e-5, 1e-5, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        e_field = e_field + 0j

        strategy = BuilderStrategyPython(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])

        shifted_field = np.zeros_like(e_field)

        for i_x_shift in range(-25,25):
            for i_y_shift in range(-25,25):
                comp = shift(e_field.real,(i_x_shift, i_y_shift),order=0) +  1j*shift(e_field.imag,(i_x_shift, i_y_shift),order=0)
                strategy.relativeShiftedCopy(i_x_shift, i_y_shift,e_field, shifted_field)

                #print(i_x_shift, i_y_shift, np.unravel_index(shifted_field.argmax(), shifted_field.shape), np.unravel_index(comp.argmax(), comp.shape), np.linalg.norm(shifted_field-comp))

                self.assertLess(np.linalg.norm(shifted_field-comp), 1e-12)

    @unittest.skip
    def testEvaluateAllIntegralDirectVsFredholmDirect(self):
        sigma_matrix = SigmaMatrixFromCovariance(xx=3e-6,
                                            yy=1e-6,
                                            xpxp=5e-6,
                                            ypyp=5e-6,
                                            xxp=1e-6,
                                            yyp=5e-7)

        x_coordinates = np.linspace(-1e-6,1e-6, 51)
        y_coordinates = np.linspace(-1e-6,1e-6, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        yy, xx = np.meshgrid(y_coordinates, x_coordinates)
        e_field = e_field + e_field * 1j * xx * yy


        strategy = BuilderStrategyPython(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])

        coordinate_vector = np.zeros((x_coordinates.size, y_coordinates.size), dtype=np.complex128)

        for i_r_x, r_x in enumerate(x_coordinates):
            if i_r_x % 4 != 0:
                continue
            print("i_x %i/%i" %(i_r_x, x_coordinates.size))
            for i_r_y, r_y in enumerate(y_coordinates):
                if i_r_y % 11 != 0:
                    continue
#                if i_r_x != 26 and i_r_y != 26:
#                    continue

                r_1 = np.array([r_x, r_y])
                coordinate_vector[:, :] = 0.0
                coordinate_vector[i_r_x, i_r_y] = 1.0

                parallel_vector = createVector(coordinate_vector.size, coordinate_vector.size)
                parallel_vector.broadcast(coordinate_vector.flatten(),root=0)

                strategy.evaluateAllR_2_Fredholm_parallel_direct(v_in=parallel_vector, v_out=parallel_vector)
                eval_fredholm = parallel_vector.fullData()
                eval_integral = strategy.evaluateAllR_2_Integral(r_1)

                diff = np.abs((eval_integral.flatten()-eval_fredholm) / eval_fredholm)
                self.assertLess(diff.max(), 1e-14)

    def testEvaluateFredholmDirectVsFredholmConvolution(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                  sigma_y=1e-6,
                                  sigma_x_prime=5e-6,
                                  sigma_y_prime=5e-6)

        x_coordinates = np.linspace(-1e-6,1e-6, 51)
        y_coordinates = np.linspace(-1e-6,1e-6, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        yy, xx = np.meshgrid(y_coordinates, x_coordinates)
        e_field = e_field + e_field * 1j * xx * yy


        strategy = BuilderStrategyPython(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])

        coordinate_vector = np.zeros((x_coordinates.size, y_coordinates.size), dtype=np.complex128)

        for i_r_x, r_x in enumerate(x_coordinates):
            if i_r_x % 4 != 0:
                continue
            print("i_x %i/%i" %(i_r_x, x_coordinates.size))
            for i_r_y, r_y in enumerate(y_coordinates):
                if i_r_y % 11 != 0:
                    continue
#                if i_r_x != 26 and i_r_y != 26:
#                    continue

                coordinate_vector[:, :] = 0.0
                coordinate_vector[i_r_x, i_r_y] = 1.0

                parallel_vector = createVector(coordinate_vector.size, coordinate_vector.size)
                parallel_vector.broadcast(coordinate_vector.flatten(), root=0)

                strategy.evaluateAllR_2_Fredholm_parallel_direct(v_in=parallel_vector, v_out=parallel_vector)
                eval_direct = parallel_vector.fullData().copy()

                parallel_vector.broadcast(coordinate_vector.flatten(),root=0)
                strategy.evaluateAllR_2_Fredholm_parallel_convolution(v_in=parallel_vector, v_out=parallel_vector)
                eval_convolution = parallel_vector.fullData().copy()

                diff = np.abs((eval_direct-eval_convolution) / eval_direct)
                self.assertLess(diff.max(), 1e-12)

    def testEvaluateFredholmConvolutionVsConvolution(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                  sigma_y=1e-6,
                                  sigma_x_prime=5e-6,
                                  sigma_y_prime=5e-6)

        x_coordinates = np.linspace(-1e-6,1e-6, 51)
        y_coordinates = np.linspace(-1e-6,1e-6, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        e_field = e_field + 0j

        yy, xx = np.meshgrid(y_coordinates, x_coordinates)
        e_field = e_field + e_field * 1j * xx * yy

        strategy = BuilderStrategyPython(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])
        strategy_convolution = BuilderStrategyConvolution(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])

        coordinate_vector = np.zeros((x_coordinates.size, y_coordinates.size), dtype=np.complex128)

        for i_r_x, r_x in enumerate(x_coordinates):
            if i_r_x % 4 != 0:
                continue
            print("i_x %i/%i" %(i_r_x, x_coordinates.size))
            for i_r_y, r_y in enumerate(y_coordinates):
                if i_r_y % 11 != 0:
                    continue
#                if i_r_x != 26 and i_r_y != 26:
#                    continue

                r_1 = np.array([r_x, r_y])
                coordinate_vector[:, :] = 0.0
                coordinate_vector[i_r_x, i_r_y] = 1.0

                parallel_vector = createVector(coordinate_vector.size, coordinate_vector.size)
                parallel_vector.broadcast(coordinate_vector.flatten(), root=0)

                strategy.evaluateAllR_2_Fredholm_parallel_convolution(v_in=parallel_vector, v_out=parallel_vector)
                eval_fredholm = parallel_vector.fullData()
                eval_integral = strategy_convolution.evaluateAllR_2_Integral(r_1)

                diff = np.abs((eval_integral.flatten()-eval_fredholm) / eval_fredholm)
                self.assertLess(diff.max(), 1e-12)


    def testEvaluateIntegralDirectVsConvolution(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                  sigma_y=1e-6,
                                  sigma_x_prime=5e-6,
                                  sigma_y_prime=5e-6)

        x_coordinates = np.linspace(-1e-6,1e-6, 51)
        y_coordinates = np.linspace(-1e-6,1e-6, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        yy, xx = np.meshgrid(y_coordinates, x_coordinates)
        e_field = e_field + e_field * 1j * xx * yy


        strategy = BuilderStrategyPython(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])
        strategy._setUpPhases()

        strategy_convolution = BuilderStrategyConvolution(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])

        for i_r_x, r_x in enumerate(x_coordinates):
            if i_r_x % 4 != 0:
                continue
            print("i_x %i/%i" %(i_r_x, x_coordinates.size))
            for i_r_y, r_y in enumerate(y_coordinates):
                if i_r_y % 11 != 0:
                    continue
#                if i_r_x != 26 and i_r_y != 26:
#                    continue

                r_1 = np.array([r_x, r_y])

                eval_fredholm = strategy.evaluateAllR_2_Integral(r_1).flatten()
                eval_conv = strategy_convolution.evaluateAllR_2_Integral(r_1).flatten()

                diff = np.abs((eval_conv-eval_fredholm) / eval_fredholm)
                self.assertLess(diff.max(), 1e-12)

    def testEvaluateCutX(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                  sigma_y=1e-6,
                                  sigma_x_prime=5e-6,
                                  sigma_y_prime=5e-6)

        x_coordinates = np.linspace(-1e-6,1e-6, 51)
        y_coordinates = np.linspace(-1e-6,1e-6, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        yy, xx = np.meshgrid(y_coordinates, x_coordinates)
        e_field = e_field + e_field * 1j * xx * yy


        strategy_convolution = BuilderStrategyConvolution(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])
        strategy_convolution.setAllCoordinates(x_coordinates, y_coordinates)

        index_y_zero = strategy_convolution.yIndexByCoordinate(0.0)

        cut = strategy_convolution.evaluateCutX()

        for i_r_x, r_x in enumerate(x_coordinates):
            if i_r_x % 4 != 0:
                continue

            print("i_x %i/%i" %(i_r_x, x_coordinates.size))
            r_1 = np.array([r_x, 0.0])
            index_x = strategy_convolution.xIndexByCoordinate(r_x)

            eval_full = strategy_convolution.evaluateAllR_2(r_1)

            for i_r_x2, r_x2 in enumerate(x_coordinates):
                index_x2 = strategy_convolution.xIndexByCoordinate(r_x2)
                eval_cut = cut[index_x, index_x2]

                diff = np.abs((eval_cut-eval_full[index_x2, index_y_zero]) / eval_full[index_x2, index_y_zero])
                self.assertLess(diff.max(), 1e-12)

    def testEvaluateCutY(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                  sigma_y=1e-6,
                                  sigma_x_prime=5e-6,
                                  sigma_y_prime=5e-6)

        x_coordinates = np.linspace(-1e-6,1e-6, 51)
        y_coordinates = np.linspace(-1e-6,1e-6, 51)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        e_field = createGaussian2D(sigma_x=1.0e-6,
                                   sigma_y=1.0e-6,
                                   x_coordinates=x_coordinates,
                                   y_coordinates=y_coordinates)
        yy, xx = np.meshgrid(y_coordinates, x_coordinates)
        e_field = e_field + e_field * 1j * xx * yy


        strategy_convolution = BuilderStrategyConvolution(x_coordinates, y_coordinates, density, x_coordinates, y_coordinates, e_field[np.newaxis,:,:])
        strategy_convolution.setAllCoordinates(x_coordinates, y_coordinates)

        index_x_zero = strategy_convolution.xIndexByCoordinate(0.0)

        cut = strategy_convolution.evaluateCutY()

        for i_r_y, r_y in enumerate(y_coordinates):
            if i_r_y % 4 != 0:
                continue

            print("i_x %i/%i" %(i_r_y, y_coordinates.size))
            r_1 = np.array([0.0, r_y])
            index_y = strategy_convolution.yIndexByCoordinate(r_y)

            eval_full = strategy_convolution.evaluateAllR_2(r_1)

            for i_r_y2, r_y2 in enumerate(y_coordinates):
                index_y2 = strategy_convolution.yIndexByCoordinate(r_y2)
                eval_cut = cut[index_y, index_y2]

                diff = np.abs((eval_cut-eval_full[index_x_zero, index_y2]) / eval_full[index_x_zero, index_y2])
                self.assertLess(diff.max(), 1e-12)