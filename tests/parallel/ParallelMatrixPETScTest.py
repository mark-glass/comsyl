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
from petsc4py import PETSc

from comsyl.parallel.ParallelVector import ParallelVector
from comsyl.parallel.ParallelMatrixPETSc import ParallelMatrixPETSc
from comsyl.math.EigenmoderStrategy import EigenmoderStartegySLEPc
from mpi4py import MPI as mpi

from tests.parallel.ParallelVectorTest import createDistributionPlan, createVector

def createMatrix(n_rows=10, n_columns=10):
    opts = PETSc.Options()

    matrix = PETSc.Mat().createDense([n_rows, n_columns])
    matrix.setFromOptions()
    matrix.setUp()

    matrix = ParallelMatrixPETSc(petsc_matrix=matrix)

    return matrix

class ParallelMatrixPETScTest(unittest.TestCase):
    def testConstructor(self):
        matrix = createMatrix(30,30)

    def testAssignment(self):
        n_rows = 231
        n_columns = 480
        matrix = createMatrix(n_rows, n_columns)

        random_matrix = np.random.random((n_rows,n_columns)) + 1j* np.random.random((n_rows,n_columns))

        for i_row in matrix.localRows():
            matrix.setRow(i_row, random_matrix[i_row, :])

        matrix.assemble()

        for i_row in matrix.localRows():
            diff = np.linalg.norm(matrix.globalRow(i_row)-random_matrix[i_row,:])
            self.assertLess(diff,1e-12)

    def testDiagonalization(self):
        n_rows = 5
        n_columns = n_rows
        number_eigenfunctions = 3

        matrix = createMatrix(n_rows, n_columns)

        for i_row in matrix.localRows():
            row = np.zeros(n_columns)
            row[i_row] = i_row
            matrix.setRow(i_row, row)

        matrix.assemble()

        eigenmoder = EigenmoderStartegySLEPc()
        eigenvalues, eigenvectors = eigenmoder.eigenfunctions(matrix,number_eigenfunctions)

        plan = matrix.distributionPlan()
        vector = ParallelVector(plan)

        for i in range(number_eigenfunctions):

            data = None
            if eigenvectors is not None:
                data = eigenvectors.globalRow(i)

            vector.broadcast(data=data, root=0)
            matrix.dot(vector)

            if eigenvectors is not None:
                accuracy = np.linalg.norm( vector.fullData() - eigenvalues[i] * eigenvectors.globalRow(i))
                self.assertLess(accuracy, 1e-12)



    @unittest.skip
    def testDot(self):
        n_rows = 10
        n_columns = n_rows
        matrix = createMatrix(n_rows, n_columns)

        for i_row in matrix.localRows():
            row = np.zeros(n_columns)
            row[i_row] = i_row
            matrix.setRow(i_row, row)

        matrix.assemble()

        plan = matrix.distributionPlan()
        vector = ParallelVector(plan)


        for i_row in range(n_rows):
            local_data = np.zeros(len(matrix.localRows()),dtype=np.complex128)
            if i_row in matrix.localRows():
                i_local_index = matrix.distributionPlan().globalToLocalIndex(i_row)
                local_data[i_local_index] = 1.0

            vector.setCollective(local_data)
            matrix.dot(vector)

            full_row = np.zeros(n_columns,dtype=np.complex128)
            full_row[i_row] = i_row
            self.assertLess(np.linalg.norm(vector.fullData()-full_row), 1e-14)