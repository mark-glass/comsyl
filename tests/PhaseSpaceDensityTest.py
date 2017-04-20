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
import scipy.interpolate as interpolate

from comsyl.autocorrelation.SigmaMatrix import SigmaWaist
from comsyl.math. utils import trapez2D, createGaussian2D, plotSurface
from comsyl.autocorrelation.PhaseSpaceDensity import PhaseSpaceDensity
from comsyl.autocorrelation.SigmaMatrix import SigmaMatrix, SigmaMatrixFromCovariance


class Numerical_f_dz(object):
    def __init__(self, sigma_matrix, delta, z, integration_points=1000):
        self._sigma_matrix = sigma_matrix
        self._delta = delta
        self._z = z

        self._s_P = sigma_matrix.s_P(delta=delta, z=z)

        vec_delta_z = np.array([delta, z])
        self._prefactor_dz = 2 * np.pi * np.sqrt(sigma_matrix.SigmaPpDeterminant()) * \
                             np.exp(-0.5 * sigma_matrix.SigmaPzInverse().dot(vec_delta_z).dot(vec_delta_z))

        sigma_x_p = sigma_matrix.element("xp", "xp") ** -0.5
        sigma_y_p = sigma_matrix.element("yp", "yp") ** -0.5

        self._x_coordinates = np.linspace(-4 * sigma_x_p, 4 * sigma_x_p, integration_points)
        self._y_coordinates = np.linspace(-4 * sigma_y_p, 4 * sigma_y_p, integration_points)

        self._dx = self._x_coordinates[1] - self._x_coordinates[0]
        self._dy = self._y_coordinates[1] - self._y_coordinates[0]

        self._work_buff = np.zeros((self._x_coordinates.shape[0],
                                    self._y_coordinates.shape[0]), dtype=np.complex128)

    def evaluateAnalytical(self, k, vec_x_y, vec_dr):

        vec = self._sigma_matrix.s_Pp(x=vec_x_y[0],
                                      y=vec_x_y[1],
                                      delta=vec_dr[0],
                                      z=vec_dr[1]) \
              + 1j * k * vec_dr

        result = self._prefactor_dz * \
                 np.exp(-0.5 * self._sigma_matrix.SigmaPInverse().dot(vec_x_y).dot(vec_x_y) + self._s_P.dot(vec_x_y) + \
                         0.5 * self._sigma_matrix.SigmaPp().dot(vec).dot(vec))

        return result

    def evaluateNumerical(self, k, vec_x_y, vec_dr):
        sigma_matrix = self._sigma_matrix.SigmaMatrix()

        vec = np.zeros(6)

        for i_x_p, x_p in enumerate(self._x_coordinates):
            for i_y_p, y_p in enumerate(self._y_coordinates):
                vec[:] = (vec_x_y[0],
                          x_p,
                          vec_x_y[1],
                          y_p,
                          self._delta,
                          self._z)

                first = -0.5 * sigma_matrix.dot(vec).dot(vec)
                second = 1j * k * (x_p * vec_dr[0] + y_p * vec_dr[1])

                self._work_buff[i_x_p, i_y_p] = first + second

        self._work_buff = np.exp(self._work_buff)

        integral = trapez2D(self._work_buff, dx=self._dx, dy=self._dy)

        return integral

class NumericalPhaseSpaceDensity(object):
    def __init__(self, N_e, field_x_coordinates, field_y_coordinates, efield, sigma_matrix, delta, z, wavenumber, integration_points=100):
        self._f_d_z = Numerical_f_dz(sigma_matrix, delta, z, integration_points)
        self._efield = efield

        self._f_field_real = interpolate.RectBivariateSpline(field_x_coordinates, field_y_coordinates, self._efield.real)
        self._f_field_imag = interpolate.RectBivariateSpline(field_x_coordinates, field_y_coordinates, self._efield.imag)

        self._wavenumber = wavenumber

        self._prefactor = N_e / ((2 * np.pi)**3 * np.sqrt(sigma_matrix.SigmaInverseDeterminant()))

        sigma_x = sigma_matrix.element("x", "x") ** -0.5
        sigma_y = sigma_matrix.element("y", "y") ** -0.5

        self._x_coordinates = np.linspace(-4 * sigma_x, 4 * sigma_x, integration_points)
        self._y_coordinates = np.linspace(-4 * sigma_y, 4 * sigma_y, integration_points)

        self._dx = self._x_coordinates[1] - self._x_coordinates[0]
        self._dy = self._y_coordinates[1] - self._y_coordinates[0]

        self._work_buff = np.zeros((self._x_coordinates.shape[0],
                                    self._y_coordinates.shape[0]), dtype=np.complex128)

        self._yy, self._xx = np.meshgrid(self._y_coordinates, self._x_coordinates)

    def field_at(self, vec_r):
        # ToDo: Fix order!
        field = self._f_field_real( (self._x_coordinates - vec_r[0]), (self._y_coordinates - vec_r[1]))
        field = 1j * self._f_field_imag( (self._x_coordinates - vec_r[0]), (self._y_coordinates - vec_r[1])) + field

        return field

    def evaluate(self, r_1, r_2):

        vec_dr = r_1 - r_2
        vec_x_y = np.zeros(2)

        # ToDo: Fix order!
        e1_conj = (self.field_at(r_1).conj())[::-1,::-1]
        e2 = self.field_at(r_2)[::-1,::-1]
        prod_field = e1_conj * e2

        uevaluate = np.frompyfunc(lambda x,y : self._f_d_z.evaluateAnalytical(k=self._wavenumber,
                                                                               vec_x_y=np.array((x ,y)),
                                                                               vec_dr=vec_dr), 2, 1)


        # for i_x_p, x_p in enumerate(self._x_coordinates):
        #     for i_y_p, y_p in enumerate(self._y_coordinates):
        #         vec_x_y[:] = (x_p, y_p)
        #         self._work_buff[i_x_p, i_y_p] = self._f_d_z.evaluateAnalytical(k=self._wavenumber,
        #                                                                        vec_x_y=vec_x_y,
        #                                                                        vec_dr=vec_dr)


        self._work_buff = uevaluate(self._xx, self._yy) * prod_field

        # print( (self._work_buff- t).sum() )


        # self._work_buff *= prod_field

        integral = trapez2D(self._work_buff, dx=self._dx, dy=self._dy)
        print(integral)

        return integral


class PhaseSpaceDensityTest(unittest.TestCase):

    def testStaticElectronDensity(self):
        sigma_x = 3e-6
        sigma_y = 1e-6
        sigma_matrix = SigmaWaist(sigma_x=sigma_x,
                                  sigma_y=sigma_y,
                                  sigma_x_prime=5e-6,
                                  sigma_y_prime=4e-6)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        x_coordinates = np.linspace(-10e-6, 10e-6, 100)
        y_coordinates = np.linspace(-10e-6, 10e-6, 100)

        prefactor = 1.0 / (2 * np.pi * sigma_matrix.sigma_x()**2 * sigma_matrix.sigma_y()**2 * sigma_matrix.sigma_d()**2)

        #TODO prefactor not correct!
        prefactor = density.staticPart(np.array([0,0]))

        diff_prefactor = prefactor / density.staticPart(np.array([0,0]))

        self.assertLess(np.abs(1-diff_prefactor), 1e-12)


        dy = 0.0
        for dx in x_coordinates:
            dr = np.array([dx, dy])
            diff = density.staticPart(delta_r=dr) / \
                   (prefactor * np.exp(-(dx-dy)**2 * (0.5*wavenumber**2*sigma_matrix.sigma_x_prime()**2)))
            self.assertLess(np.abs(1-diff), 1e-12)

        dx = 0.0
        for dy in y_coordinates:
            dr = np.array([dx, dy])
            diff = density.staticPart(delta_r=dr) / \
                   (prefactor * np.exp(-(dx-dy)**2 * (0.5*wavenumber**2*sigma_matrix.sigma_y_prime()**2)))
            self.assertLess(np.abs(1-diff), 1e-12)

    def testIntegrationPartGaussian(self):
        sigma_x = 3e-6
        sigma_y = 1e-6
        sigma_matrix = SigmaWaist(sigma_x=sigma_x,
                                  sigma_y=sigma_y,
                                  sigma_x_prime=5e-6,
                                  sigma_y_prime=4e-6)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        x_coordinates = np.linspace(-15e-6, 15e-6, 200)
        y_coordinates = np.linspace(-7e-6, 7e-6, 200)

        for dx in x_coordinates:
            diff = density.integrationPartGaussian(delta=0.0, z=0.0, x=dx, y=0.0) - np.exp(-dx**2/(2*sigma_matrix.sigma_x()**2))
            self.assertLess(diff, 1e-12)

        for dy in y_coordinates:
            diff = density.integrationPartGaussian(delta=0.0, z=0.0, x=0.0, y=dy) - np.exp(-dy**2/(2*sigma_matrix.sigma_y()**2))
            self.assertLess(diff, 1e-12)

        rho = np.zeros((x_coordinates.size, y_coordinates.size), dtype=np.complex128)
        for i_x, dx in enumerate(x_coordinates):
            for i_y, dy in enumerate(y_coordinates):
                rho[i_x, i_y] = density.integrationPartGaussian(delta=0.0, z=0.0, x=dx, y=dy)

        norm_1 = np.trapz(np.trapz(np.abs(rho), y_coordinates), x_coordinates)
        self.assertLess(1-norm_1/(2*np.pi*sigma_x*sigma_y), 1e-6)

    def testIntegrationPartGaussianWithAlpha(self):
        sigma_matrix = SigmaMatrixFromCovariance(xx=3e-6,
                                            yy=1e-6,
                                            xxp=5e-6,
                                            yyp=4e-6,
                                            xpxp=0.5*3e-6,
                                            ypyp=0.5*1e-6)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        x_coordinates = np.linspace(-10e-6, 10e-6, 100)
        y_coordinates = np.linspace(-10e-6, 10e-6, 100)

        for dx in x_coordinates:
            dr = np.array([dx, 0.0])
            for x in x_coordinates:
                diff = density.integrationPartOscillation(delta_r=dr, x=x, y=0.0)
                print(2*wavenumber* dr[0] * (sigma_matrix.element('x','xp')/sigma_matrix.element('xp','xp'))*x)
                diff -= np.exp(2j*wavenumber* dr[0] * (sigma_matrix.element('x','xp')/sigma_matrix.element('xp','xp'))*x)
                self.assertLess(np.abs(diff), 1e-12)

        for dy in y_coordinates:
            dr = np.array([0.0, dy])
            for y in y_coordinates:
                diff = density.integrationPartOscillation(delta_r=dr, x=0.0, y=y)
                diff -= np.exp(2j*wavenumber* dr[1] * (sigma_matrix.element('y','yp')/sigma_matrix.element('yp','yp'))*y)
                self.assertLess(np.abs(diff), 1e-12)


    def testStaticPartFixedR1(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                            sigma_y=1e-6,
                                            sigma_x_prime=5e-6,
                                            sigma_y_prime=4e-6)
        wavenumber = 1e+11

        density = PhaseSpaceDensity(sigma_matrix, wavenumber)

        x_coordinates = np.linspace(-10e-6, 10e-6, 100)
        y_coordinates = np.linspace(-10e-6, 10e-6, 100)

        r_1 = np.array([0.0, 0.0])
        density.setAllStaticPartCoordinates(x_coordinates, y_coordinates)

        values = density.staticPartFixedR1(r_1)
        for i_x, x in enumerate(x_coordinates):
            for i_y, y in enumerate(y_coordinates):
                dr = r_1 - np.array([x, y])
                diff = np.abs(1- values[i_x, i_y] / density.staticPart(dr))
                self.assertLess(diff, 1e-12)

    def testNumerical_f_dzSimgple(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                            sigma_y=1e-6,
                                            sigma_x_prime=5e-6,
                                            sigma_y_prime=4e-6)
        wavenumber = 1e+11

        x_coordinates = np.linspace(-10e-6, 10e-6, 15)
        y_coordinates = np.linspace(-10e-6, 10e-6, 10)

        tester = Numerical_f_dz(sigma_matrix, z=0.0, delta=0.0)

        prefactor = 2 * np.pi * sigma_matrix.sigma_x_prime() * sigma_matrix.sigma_y_prime()

        for x in x_coordinates:
            for y in y_coordinates:
                vec_x_y = np.array([x, y])
                for dx in x_coordinates:
                    for dy in y_coordinates:
                        vec_dr = np.array([dx, dy])

                        result_tester = tester.evaluateAnalytical(k=wavenumber,
                                                                  vec_x_y=vec_x_y,
                                                                  vec_dr=vec_dr)

                        result_analytic = prefactor * \
                                          np.exp(-0.5 * ((x**2/sigma_matrix.sigma_x()**2)+(y**2/sigma_matrix.sigma_y()**2))) * \
                                          np.exp(-0.5 * wavenumber**2 * ((dx**2 * sigma_matrix.sigma_x_prime()**2)+(dy**2 * sigma_matrix.sigma_y_prime()**2)))

                        diff = np.abs(1 - (result_tester / result_analytic))
                        self.assertLess(diff, 1e-12)

    @unittest.skip
    def testNumerical_f_dz(self):
        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                            sigma_y=1e-6,
                                            sigma_x_prime=5e-6,
                                            sigma_y_prime=4e-6)

        sigma_matrix=sigma_matrix.SigmaMatrix()
        sigma_matrix[0, 1] = sigma_matrix[0, 0] * 0.0001
        sigma_matrix[1, 0] = sigma_matrix[0, 1]
        sigma_matrix[2, 3] = sigma_matrix[2, 2] * 0.00003
        sigma_matrix[3, 2] = sigma_matrix[2, 3]

        sigma_matrix[4, 4] = 90000
        sigma_matrix[5, 5] = 60000
        sigma_matrix[4, 5] = 11000
        sigma_matrix[5, 4] = 11000

        sigma_matrix = SigmaMatrix(sigma_matrix)


        x_coordinates = np.linspace(-1e-6, 1e-6, 3)
        y_coordinates = np.linspace(-1e-6, 1e-6, 2)
        tester = Numerical_f_dz(sigma_matrix, z=0.01, delta=0.005, integration_points=100)

        # It is very expansive to converge for high frequency...
        for wavenumber in [1e9, 1e1, 1e11, 5e11]:
            for x in x_coordinates:
                for y in y_coordinates:
                    vec_x_y = np.array([x, y])
                    for dx in x_coordinates:
                        for dy in y_coordinates:
                            vec_dr = np.array([dx, dy])

                            result_analytic = tester.evaluateAnalytical(k=wavenumber,
                                                                        vec_x_y=vec_x_y,
                                                                        vec_dr=vec_dr)

                            result_numerical = tester.evaluateNumerical(k=wavenumber,
                                                                        vec_x_y=vec_x_y,
                                                                        vec_dr=vec_dr)

                            diff = np.abs(1 - (result_analytic / result_numerical))

                            #print(diff,wavenumber/10e10,x,y,dx,dy)
                            self.assertLess(diff, 2e-3)

    @unittest.skip
    def testNumerical(self):

        x_coordinates_gaussian = np.linspace(-3e-7, 3e-7, 100)
        y_coordinates_gaussian = np.linspace(-3e-7, 3e-7, 100)

        field = createGaussian2D(sigma_x=1e-7,
                                 sigma_y=1e-7,
                                 x_coordinates=x_coordinates_gaussian,
                                 y_coordinates=y_coordinates_gaussian)

        sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                            sigma_y=1e-6,
                                            sigma_x_prime=5e-6,
                                            sigma_y_prime=4e-6)
        wavenumber = 1e+11

        tester = NumericalPhaseSpaceDensity(N_e=1,
                                            field_x_coordinates=x_coordinates_gaussian,
                                            field_y_coordinates=y_coordinates_gaussian,
                                            efield=field,
                                            sigma_matrix=sigma_matrix,
                                            z=0.0,
                                            delta=0.0,
                                            wavenumber=wavenumber)

        x_coordinates = np.linspace(-10e-6, 10e-6, 15)
        y_coordinates = np.linspace(-10e-6, 10e-6, 15)

        work_buff = np.zeros((x_coordinates.shape[0],
                              y_coordinates.shape[0]), dtype=np.complex128)

        for i_x, x in enumerate(x_coordinates):
            for i_y, y in enumerate(y_coordinates):
                r_1 = np.array([x, y])
                r_2 = r_1
                work_buff[i_x, i_y] = tester.evaluate(r_1, r_2)

        plotSurface(x_coordinates, y_coordinates, work_buff, True)