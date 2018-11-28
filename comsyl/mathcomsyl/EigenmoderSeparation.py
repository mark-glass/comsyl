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
from comsyl.utils.Logger import log
from comsyl.mathcomsyl.Eigenmoder import Eigenmoder
from comsyl.mathcomsyl.MatrixBuilder import MatrixBuilder
from comsyl.mathcomsyl.utils import sorted2DIndices, norm2D
from comsyl.mathcomsyl.Twoform import Twoform
from comsyl.mathcomsyl.TwoformVectors import TwoformVectorsEigenvectors

class EigenmoderSeparation(object):
    def __init__(self, x_coordinates, y_coordinates, cut_x, cut_y):

        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates

        self.cut_x = cut_x
        self.cut_y = cut_y

    def _prepare(self, number_horizontal_modes, number_vertical_modes):
        x_coordinates = self._x_coordinates
        y_coordinates = self._y_coordinates

        x_coordinate = 0.0
        eigenmoder = Eigenmoder(np.array([x_coordinate]), y_coordinates)

        builder = MatrixBuilder(np.array([x_coordinate]), y_coordinates)
        builder._mode_element_wise = True
        work_matrix = builder._createParallelMatrix(self.evaluate)
        self._twoform_y = eigenmoder.eigenmodes(work_matrix, number_modes=number_vertical_modes)

        y_coordinate = 0.0
        eigenmoder = Eigenmoder(x_coordinates, np.array([y_coordinate]))

        builder = MatrixBuilder(x_coordinates, np.array([y_coordinate]))
        builder._mode_element_wise = True
        work_matrix = builder._createParallelMatrix(self.evaluate)
        self._twoform_x = eigenmoder.eigenmodes(work_matrix, number_modes=number_horizontal_modes)

    def determineReconstructionQuality(self, number_modes=None):
        x_coordinates = self._x_coordinates
        y_coordinates = self._y_coordinates

        # for x1 in x_coordinates[::10]:
        #     for x2 in x_coordinates[::10]:
        #         r1 = np.array((x1, 0))
        #         r2 = np.array((x2, 0))
        #
        #         i_x1 = np.abs(x_coordinates-x1).argmin()
        #         i_x2 = np.abs(x_coordinates-x2).argmin()
        #
        #         e1 = self.cut_x[i_x1, i_x2]
        #         e2 = self._twoform_x.evaluate(r1,r2).conj()
        #         print(e1,e2, np.abs((e1-e2)/e1))
        #
        # for y1 in y_coordinates[::10]:
        #     for y2 in y_coordinates[::10]:
        #         r1 = np.array((0, y1))
        #         r2 = np.array((0, y2))
        #
        #         i_y1 = np.abs(y_coordinates-y1).argmin()
        #         i_y2 = np.abs(y_coordinates-y2).argmin()
        #
        #         e1 = self.cut_y[i_y1, i_y2]
        #         e2 = self._twoform_y.evaluate(r1, r2).conj()
        #         print(e1,e2, np.abs((e1-e2)/e1))

        # for x1 in x_coordinates[::65]:
        #     for x2 in x_coordinates[::60]:
        #         r1 = np.array((x1, 0))
        #         r2 = np.array((x2, 0))
        #         e1 = af.evaluate(r1, r2)
        #         e2 = self.evaluate(r1, r2) #self._twoform_x.evaluate(r_x1,r_x2).conj() * self._twoform_y.evaluate(r_y1, r_y2).conj() / self._gamma_at_0
        #         print(e1,e2, np.abs((e1-e2)/e1))
        #
        # print("------------------------")
        #
        # for y1 in y_coordinates[::50]:
        #     for y2 in y_coordinates[::50]:
        #         r1 = np.array((0, y1))
        #         r2 = np.array((0, y2))
        #         e1 = af.evaluate(r1, r2)
        #         e2 = self.evaluate(r1, r2) #self._twoform_x.evaluate(r_x1,r_x2).conj() * self._twoform_y.evaluate(r_y1, r_y2).conj() / self._gamma_at_0
        #         print(e1,e2, np.abs((e1-e2)/e1))
        #
        # print("------------------------")
        #
        # for x1 in x_coordinates[::90]:
        #     for x2 in x_coordinates[::75]:
        #         for y1 in y_coordinates[::60]:
        #             for y2 in y_coordinates[::80]:
        #                 r1 = np.array((x1, y1))
        #                 r2 = np.array((x2, y2))
        #                 e1 = af.evaluate(r1, r2)
        #                 e2 = self.evaluate(r1, r2) #self._twoform_x.evaluate(r_x1,r_x2).conj() * self._twoform_y.evaluate(r_y1, r_y2).conj() / self._gamma_at_0
        #                 print(e1,e2, np.abs((e1-e2)/e1))


        # for x1 in x_coordinates[::10]:
        #     for x2 in x_coordinates[::10]:
        #         r1 = np.array((x1, 0))
        #         r2 = np.array((x2, 0))
        #
        #         e1 = af.evaluate(r1, r2)
        #         e2 = self.evaluate(r1, r2)
        #         print(e1,e2, np.abs((e1-e2)/e1))
        #
        # for y1 in y_coordinates[::10]:
        #     for y2 in y_coordinates[::10]:
        #         r1 = np.array((0, y1))
        #         r2 = np.array((0, y2))
        #
        #         e1 = af.evaluate(r1, r2)
        #         e2 = self.evaluate(r1, r2)
        #         print(e1,e2, np.abs((e1-e2)/e1))

        cut_x, tmp = self._twoform_x.XYcuts()
        tmp, cut_y = self._twoform_y.XYcuts()

        diff_norm_cut_x = norm2D(x_coordinates, x_coordinates, self.cut_x - cut_x.conj())/norm2D(x_coordinates, x_coordinates, self.cut_x)
        diff_norm_cut_y = norm2D(y_coordinates, y_coordinates, self.cut_y - cut_y.conj())/norm2D(y_coordinates, y_coordinates, self.cut_y)

        twoform = self.twoform(int(number_modes))
        cut_x, cut_y = twoform.XYcuts()
        diff_norm_cut_x2 = norm2D(x_coordinates, x_coordinates, self.cut_x - cut_x)/norm2D(x_coordinates, x_coordinates, self.cut_x)
        diff_norm_cut_y2 = norm2D(y_coordinates, y_coordinates, self.cut_y - cut_y)/norm2D(y_coordinates, y_coordinates, self.cut_y)

        return diff_norm_cut_x, diff_norm_cut_y, diff_norm_cut_x2, diff_norm_cut_y2

    def xIndexByCoordinate(self, x):
        return np.abs(self._x_coordinates-x).argmin() - int(self._x_coordinates.shape[0]/2)
        return np.abs(self._x_coordinates-x).argmin()

    def yIndexByCoordinate(self, y):
        return np.abs(self._y_coordinates-y).argmin() - int(self._y_coordinates.shape[0]/2)
        return np.abs(self._y_coordinates-y).argmin()

    def evaluate_cut_y(self, r_1, r_2):
        i_y_1 = self.yIndexByCoordinate(r_1[1])
        i_y_2 = self.yIndexByCoordinate(r_2[1])

        res = self.cut_y[i_y_1, i_y_2]

        return res

    def evaluate(self, r_1, r_2):
        i_x_1 = self.xIndexByCoordinate(r_1[0])
        i_y_1 = self.yIndexByCoordinate(r_1[1])

        i_x_2 = self.xIndexByCoordinate(r_2[0])
        i_y_2 = self.yIndexByCoordinate(r_2[1])

        i_x_zero = self.xIndexByCoordinate(0.0)
        i_y_zero = self.yIndexByCoordinate(0.0)

        res = self.cut_x[i_x_1, i_x_2] * self.cut_y[i_y_1, i_y_2]
        res /= (self.cut_x[i_x_zero, i_x_zero] * self.cut_y[i_y_zero, i_y_zero]) ** 0.5

        return res

    def twoform(self, number_eigenvalues=50):
        i_x_zero = self.xIndexByCoordinate(0.0)
        i_y_zero = self.yIndexByCoordinate(0.0)
        self._prepare(number_horizontal_modes=number_eigenvalues, number_vertical_modes=100)

        twoform_x = self._twoform_x
        twoform_y = self._twoform_y

        eigenvalues_x = twoform_x.eigenvalues()[:number_eigenvalues]
        eigenvalues_y = twoform_y.eigenvalues()[:number_eigenvalues]

        sorted_indices = sorted2DIndices(eigenvalues_x, eigenvalues_y)

        eigenvectors = np.zeros((number_eigenvalues, self._x_coordinates.size, self._y_coordinates.size), dtype=np.complex128)
        eigenvalues = np.zeros(number_eigenvalues, dtype=np.complex128)
        for i_mode in range(number_eigenvalues):
            i_vector_x, i_vector_y = sorted_indices[i_mode]
            vector_x = twoform_x.vector(i_vector_x)[:, 0]
            vector_y = twoform_y.vector(i_vector_y)[0, :].conj()
            eigenvectors[i_mode, :, :] = np.outer(vector_x, vector_y)
            eigenvalues[i_mode] = eigenvalues_x[i_vector_x] * eigenvalues_y[i_vector_y]

        # Remove: Multiply by dV
        dV = norm2D(self._x_coordinates, self._y_coordinates, eigenvectors[0, :, :])
        eigenvectors[:,:,:] /= dV
        eigenvalues *= dV**2 / (self.cut_x[i_x_zero, i_x_zero] * self.cut_y[i_y_zero, i_y_zero]) ** 0.5

        twoform = Twoform(self._x_coordinates, self._y_coordinates, None, eigenvalues, TwoformVectorsEigenvectors(eigenvectors))
        return twoform