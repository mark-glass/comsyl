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
from petsc4py import PETSc


from comsyl.parallel.utils import barrier
from comsyl.parallel.DistributionPlan import DistributionPlan
from comsyl.parallel.ParallelMatrix import ParallelMatrix
from comsyl.parallel.ParallelMatrixPETSc import ParallelMatrixPETSc

from comsyl.utils.Logger import log, logProgress


class MatrixBuilder(object):
    def __init__(self, coordinates_x, coordinates_y):
        self._coordinates_x = coordinates_x
        self._coordinates_y = coordinates_y
        self._mode_element_wise = False

    def productCoordinates(self):
        product_coordinates = np.zeros((len(self._coordinates_x)*len(self._coordinates_y),2))

        i_c = 0
        for x in self._coordinates_x:
            for y in self._coordinates_y:
                product_coordinates[i_c, 0] = x
                product_coordinates[i_c, 1] = y
                i_c += 1

        return product_coordinates

    def _createNumpyMatrix(self, f_gamma):
        product_coordinates=self.productCoordinates()
        work_matrix = np.zeros( (product_coordinates.shape[0],product_coordinates.shape[0]),dtype=np.complex128)

        product_coordinates=self.productCoordinates()

        n_coordinates = product_coordinates.shape[0]
        for i in range(n_coordinates):
            self._printProgress(n_coordinates, i)

            r_i = product_coordinates[i, :]
            for j in range(n_coordinates):
                r_j = product_coordinates[j, :]

                work_matrix[i, j] = f_gamma(r_i, r_j)

        return work_matrix

    def _createParallelMatrixPETSc(self, f_gamma):
        product_coordinates=self.productCoordinates()

        n_coordinates = product_coordinates.shape[0]

        n_rows=product_coordinates.shape[0]
        n_columns=product_coordinates.shape[0]
        petsc_matrix = PETSc.Mat().createDense([n_rows, n_columns])
        petsc_matrix.setUp()

        matrix = ParallelMatrixPETSc(petsc_matrix)
        distribution_plan = matrix.distributionPlan()

        if self._mode_element_wise:
            for i_row in distribution_plan.localRows():

                self._printProgress(n_coordinates, i_row)

                r_i = product_coordinates[i_row, :]
                for i_column in range(n_coordinates):

                    r_j = product_coordinates[i_column, :]
                    value = f_gamma(r_i, r_j)

                    # TODO
                    #raise NotImplementedError("Can only handle entire rows")
                    petsc_matrix[i_row, i_column] = value

        else:
            for i_row in distribution_plan.localRows():
                self._printProgress(len(distribution_plan.localRows()), i_row)

                r_i = product_coordinates[i_row, :]
                value = f_gamma(r_i)
                value = value.reshape(value.size).conj()

                matrix.setRow(global_index=i_row,
                              content=value)

        log("Waiting for others")
        barrier()
        log("done")

        log("PETSc matrix assembling")
        matrix.assemble()
#        matrix.transpose()
        log("done")
        return matrix


    def _createParallelMatrix(self, f_gamma):
        log("Building matrix")
        return self._createParallelMatrixPETSc(f_gamma)

        product_coordinates=self.productCoordinates()

        n_coordinates = product_coordinates.shape[0]

        distribution_plan = DistributionPlan(communicator=mpi.COMM_WORLD,
                                             n_rows=product_coordinates.shape[0],
                                             n_columns=product_coordinates.shape[0])

        matrix = ParallelMatrix(distribution_plan=distribution_plan)

        if self._mode_element_wise:
            for i_row in distribution_plan.localRows():

                self._printProgress(n_coordinates, i_row)

                r_i = product_coordinates[i_row, :]
                for i_column in range(n_coordinates):

                    r_j = product_coordinates[i_column, :]
                    value = f_gamma(r_i, r_j)

                    # TODO
                    raise NotImplementedError("Can only handle entire rows")
                    # matrix.setElement(i_row, i_column, value)

        else:
            for i_row in distribution_plan.localRows():
                self._printProgress(len(distribution_plan.localRows()), i_row)

                r_i = product_coordinates[i_row, :]
                value = f_gamma(r_i)
                value = value.reshape(value.size)

                matrix.setRow(global_index=i_row,
                              content=value)

        if distribution_plan.communicator().Get_rank() == 0:
            log("done")

        return matrix

    def _printProgress(self, n_coordinates, index):
        logProgress(n_coordinates, index, "Matrix building")
