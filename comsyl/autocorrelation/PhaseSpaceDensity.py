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



__author__ = 'mglass'

import numpy as np

# TODO: consider couplings to delta during creation of weighted fields
# TODO: normalization for delta only right for uncoupled delta
class PhaseSpaceDensity(object):
    def __init__(self, sigma_matrix, k, number_electrons=1):
        self._number_electrons = number_electrons
        self._sigma_matrix = sigma_matrix
        self._k = k

        sigma = self._sigma_matrix
        SPp = sigma.SPp()
        self._gaussian_matrix = sigma.SigmaPInverse()
        self._gaussian_matrix2 = SPp.transpose().dot(sigma.SigmaPp().dot(SPp))

        self._setStaticPartPrefactor()

    def s_P_p_d_z_r_array(self, x_coordinates, y_coordinates):

        yy, xx = np.meshgrid(y_coordinates, x_coordinates)
        s_P_p_d_z_r_array = (1j * self._k * xx, 1j * self._k * yy)
        return s_P_p_d_z_r_array

    def _setStaticPartPrefactor(self):
        # normalization for delta only right for uncoupled delta and is considered during field weighting
        pre_factor = self._number_electrons
        pre_factor /= (2*np.pi * self._sigma_matrix.moment("x", "x"))**0.5
        pre_factor /= (2*np.pi * self._sigma_matrix.moment("y", "y"))**0.5

        self._static_part_prefactor = pre_factor
        self._sigma_Pp = self._sigma_matrix.SigmaPp()

    def staticPart(self, delta_r):

        exp_Pp = np.exp(-0.5 * self._k**2 * (self._sigma_matrix.SigmaPp().dot(delta_r).dot(delta_r) ))

        res = self._static_part_prefactor * exp_Pp
        return res

    def setAllStaticPartCoordinates(self, x_coordinates, y_coordinates):
        self._all_coordiantes_x = np.array(x_coordinates).copy()
        self._all_coordiantes_y = np.array(y_coordinates).copy()

        self._work_r = np.array([0.0,0.0], dtype=np.complex128)
        self._work_result = np.zeros((x_coordinates.shape[0],
                                      y_coordinates.shape[0]), dtype=np.complex128)

    def staticPartFixedR1(self, r_1):

        s_P_p_d_z_r_array = self.s_P_p_d_z_r_array(x_coordinates=r_1[0]-self._all_coordiantes_x,
                                                   y_coordinates=r_1[1]-self._all_coordiantes_y)

        exponent = self._sigma_Pp[0, 0] * s_P_p_d_z_r_array[0] ** 2 + \
                   (self._sigma_Pp[0, 1] + self._sigma_Pp[1, 0]) * s_P_p_d_z_r_array[0] * s_P_p_d_z_r_array[1] + \
                    self._sigma_Pp[1, 1] * s_P_p_d_z_r_array[1] ** 2

        result = self._static_part_prefactor * np.exp(0.5 * exponent)

        return result

    def integrationPartGaussian(self, delta, z, x, y):

        xy = np.array([x, y])

        res = np.exp(-0.5 * self._gaussian_matrix.dot(xy).dot(xy))
        res *= np.exp(0.5 * self._gaussian_matrix2.dot(xy).dot(xy))

        # TODO: FIX!
        #vec = self._sigma_matrix.s_P(delta, z).transpose()
        #vec += self._sigma_matrix.s_P_p_d_z(delta, z).transpose().dot(self._sigma_matrix.SigmaPp().dot(self._sigma_matrix.SPp()))

        #res *= np.exp(vec.dot(xy))
        #print(res[0,0],x,y)
        return res

    def expPhaseMatrix(self):
        res = -1j * self._k * self._sigma_matrix.SigmaPp().dot(self._sigma_matrix.SPp())
        return res

    def expPhasePointsPerOscillation(self, delta_r):
        res = self.expPhaseMatrix().dot(delta_r) / (2*np.pi)

        return res.imag

    def integrationPartOscillation(self, delta_r, x, y):
        xy = np.array([[x],
                       [y]])

        phase_matrix = self.expPhaseMatrix()
        res = np.exp(delta_r.transpose().dot(phase_matrix.dot(xy)))
        return res

    def spatialPartSigmaX(self):
        return self._gaussian_matrix[0, 0] ** -0.5

    def spatialPartSigmaY(self):
        return self._gaussian_matrix[1, 1] ** -0.5

    def divergencePartSigmaX(self):
        return (self._k**2 * self._sigma_Pp[0, 0]) ** -0.5

    def divergencePartSigmaY(self):
        return (self._k**2 * self._sigma_Pp[1, 1]) ** -0.5

    def normalizationConstant(self):
        return self.staticPart(np.array([0.0, 0.0]))

    def isAlphaZero(self):
        return self._sigma_matrix.isAlphaZero()
