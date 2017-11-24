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

# What is called here SigmaMatrix is in fact \Sigma^{-1} in thesis Eq 2.10
#TODO rename to SigmaMatrixInverse or MMatrix
class SigmaMatrix(object):
    def __init__(self, sigma_matrix):
        coordinates = ['x','xp','y','yp','d','z']
        #TODO: check positive definite
        # Test if sigma_matrix is symmetric
        if np.linalg.norm(sigma_matrix-sigma_matrix.transpose()) > sigma_matrix.max() * 1e-12:
            raise Exception("Sigma matrix is not symmetric!")

        self._sigma_matrix = sigma_matrix.copy()
        self._covariance_matrix = np.linalg.inv(self._sigma_matrix)
        self._sigma = {}
        self._moment = {}

        for c_1 in coordinates:
            self._sigma[c_1] = {}
            self._moment[c_1] = {}

        for i_c_1, c_1 in enumerate(coordinates):
            for i_c_2, c_2 in enumerate(coordinates):
                self._sigma[c_1][c_2] = sigma_matrix[i_c_1, i_c_2]
                self._moment[c_1][c_2] = self._covariance_matrix[i_c_1, i_c_2]

        self._setStaticMatrices()
        self._setStaticDeterminants()

    def SigmaMatrix(self):
        return self._sigma_matrix

    def element(self, i, j):
        return self._sigma[i][j]

    def moment(self, i, j):
        return self._moment[i][j]
        

    def _setStaticMatrices(self):
        self._setSPp()
        self._setSigmaPInverse()
        self._setSigmaPzInverse()
        self._setSigmaPp()

    def _setStaticDeterminants(self):
        self._det_SPp = np.linalg.det(self.SPp())
        self._det_SigmaPInverse = np.linalg.det(self.SigmaPInverse())
        self._det_SigmaPzInverse = np.linalg.det(self.SigmaPzInverse())
        self._det_SigmaPp = np.linalg.det(self.SigmaPp())
        self._det_SigmaMatrix = np.linalg.det(self._sigma_matrix)

    def SigmaInverseDeterminant(self):
        return self._det_SigmaMatrix

    def s_P(self, delta, z):
        sigma = self._sigma

        s_P = 2 * np.array([sigma['x']['d'] * delta + sigma['x']['z'] * z,
                            sigma['y']['d'] * delta + sigma['y']['z'] * z])
        return s_P

    def s_Pp_x_y(self, x, y):
        sigma = self._sigma

        s_Pp_x_y = 2 * np.array([sigma['xp']['x'] * x + sigma['xp']['y'] * y,
                                 sigma['yp']['x'] * x + sigma['yp']['y'] * y])
        return s_Pp_x_y

    def s_Pp_d_z(self, delta, z):
        sigma = self._sigma

        s_Pp_d_z = 2 * np.array([sigma['xp']['d'] * delta + sigma['xp']['z'] * z,
                                 sigma['yp']['d'] * delta + sigma['yp']['z'] * z])
        return s_Pp_d_z

    def s_Pp(self, x, y, delta, z):
        return self.s_Pp_x_y(x, y) + self.s_Pp_d_z(delta, z)

    def _setSPp(self):
        sigma = self._sigma
        S_P_p = np.array([[sigma['xp']['x'], sigma['xp']['y']],
                          [sigma['yp']['x'], sigma['yp']['y']]])
        self._S_P_p = S_P_p

    def SPp(self):
        return self._S_P_p

    def SPpDeterminant(self):
        return self._det_SPp

    def _setSigmaPInverse(self):
        sigma = self._sigma
        sigma_P_inv = np.array([[sigma['x']['x'], sigma['x']['y']],
                                [sigma['y']['x'], sigma['y']['y']]])

        self._sigma_P_inv = sigma_P_inv

    def SigmaPInverse(self):
        return self._sigma_P_inv

    def SigmaPInverseDeterminant(self):
        return self._det_SigmaPInverse

    def _setSigmaPzInverse(self):
        sigma = self._sigma
        sigma_P_z_inv = np.array([[sigma['d']['d'], sigma['d']['z']],
                                  [sigma['z']['d'], sigma['z']['z']]])
        self._sigma_P_z_inv = sigma_P_z_inv

    def SigmaPzInverse(self):
        return self._sigma_P_z_inv

    def SigmaPzInverseDeterminant(self):
        return self._det_SigmaPzInverse

    def _setSigmaPp(self):
        sigma = self._sigma
        sigma_P_p = (1.0/(sigma['xp']['xp']*sigma['yp']['yp']-sigma['xp']['yp'] * sigma['yp']['xp'])) * \
                    np.array([[sigma['yp']['yp'], -sigma['yp']['xp']],
                              [-sigma['xp']['yp'], sigma['xp']['xp']]])

        self._sigma_P_p = sigma_P_p

    def SigmaPp(self):
        return self._sigma_P_p

    def SigmaPpDeterminant(self):
        return self._det_SigmaPp

    def isAlphaZero(self):
        e_x = self.moment("x","xp")
        e_y = self.moment("y","yp")

        if np.abs(e_x) < 1e-80 and np.abs(e_y) < 1e-80:
            return True

        return False

    def asNumpyArray(self):
        return self._sigma_matrix

    @staticmethod
    def fromNumpyArray(array):
        return SigmaMatrix(array)

    def copy(self):
        return SigmaMatrix(self._sigma_matrix)


class SigmaMatrixFromCovariance(SigmaMatrix):
    def __init__(self, xx, xxp, xpxp, yy, yyp, ypyp, sigma_dd=1e-59):
        sigma_matrix = np.zeros((6, 6))
        covariance_matrix = np.zeros((4, 4))

        covariance_matrix[0, 0] = xx
        covariance_matrix[0, 1] = xxp
        covariance_matrix[1, 1] = xpxp
        covariance_matrix[1, 0] = xxp

        covariance_matrix[2, 2] = yy
        covariance_matrix[2, 3] = yyp
        covariance_matrix[3, 3] = ypyp
        covariance_matrix[3, 2] = yyp

        sigma_matrix[0:4, 0:4] = np.linalg.inv(covariance_matrix)


        if sigma_dd > 1e-60:
            sigma_matrix[4, 4] = sigma_dd ** -2

        sigma_matrix[5, 5] = np.sqrt(2 * np.pi)

        #print(covariance_matrix - np.linalg.inv(sigma_matrix)[0:4, 0:4])


        SigmaMatrix.__init__(self, sigma_matrix)

    def sigma_x(self):
        return self._sigma['x']['x'] ** -0.5

    def sigma_y(self):
        return self._sigma['y']['y'] ** -0.5

    def sigma_x_prime(self):
        return self._sigma['xp']['xp'] ** -0.5

    def sigma_y_prime(self):
        return self._sigma['yp']['yp'] ** -0.5

    def sigma_d(self):
        return self._sigma['d']['d'] ** -0.5

    def sigma_x_xprime(self):
        return self._sigma['x']['xp'] ** -0.5

    def sigma_y_yprime(self):
        return self._sigma['y']['yp'] ** -0.5


class SigmaWaist(SigmaMatrixFromCovariance):
    def __init__(self, sigma_x, sigma_y, sigma_x_prime, sigma_y_prime, sigma_dd=1e-59):

        xx = sigma_x ** 2
        xpxp = sigma_x_prime ** 2
        yy = sigma_y ** 2
        ypyp = sigma_y_prime ** 2

        SigmaMatrixFromCovariance.__init__(self,
                                           xx=xx,
                                           xxp=0.0,
                                           xpxp=xpxp,
                                           yy=yy,
                                           yyp=0.0,
                                           ypyp=ypyp,
                                           sigma_dd=sigma_dd)