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

from comsyl.parallel.DistributionPlan import DistributionPlan
from comsyl.parallel.ParallelMatrix import ParallelMatrix
from mpi4py import MPI as mpi

from tests.parallel.ParallelVectorTest import createDistributionPlan, createVector

def createMatrix(rows=10, columns=10):
    plan = createDistributionPlan(rows, columns)
    matrix = ParallelMatrix(distribution_plan=plan)

    return matrix

class ParallelMatrixTest(unittest.TestCase):
    def testLocalShape(self):
        matrix = createMatrix()
        local_shape = matrix.localShape()
        n_rows = matrix.distributionPlan().localShape()[0]
        self.assertEqual(local_shape, (n_rows, 10))

    def testLocalRows(self):
        n_rows = 22
        matrix = createMatrix(rows=n_rows)

        if matrix.communicator().Get_size() > 1:
            return

        local_rows = matrix.localRows()
        self.assertLess(np.linalg.norm(local_rows-np.arange(n_rows)), 1e-10)

    def testLocalColumns(self):
        n_columns = 22
        matrix = createMatrix(columns=n_columns)

        if matrix.communicator().Get_size() > 1:
            return

        local_columns = matrix.localColumns()
        self.assertLess(np.linalg.norm(local_columns-np.arange(n_columns)), 1e-10)

    def testSetRow(self):

        n_columns = 22
        matrix=createMatrix(rows=10, columns=n_columns)

        local_index = 1
        global_index = matrix.distributionPlan().localToGlobalIndex(local_index)
        content = np.random.random(n_columns)

        matrix.setRow(global_index, content)
        local_matrix = matrix.localMatrix()

        self.assertLess(np.linalg.norm(local_matrix[local_index, :] - content), 1e-10)


    def testDot(self):

        n_rows = 700
        n_columns = n_rows

        parallel_vector = createVector(rows=n_rows, columns=n_columns)
        matrix = ParallelMatrix(parallel_vector.distributionPlan())

        communicator = matrix.distributionPlan().communicator()

        if communicator.Get_rank() == 0:
            entire_vector = np.array(np.random.random(n_columns), dtype=np.complex128)
            entire_vector /= np.linalg.norm(entire_vector)

            entire_matrix = np.random.random((n_rows, n_columns))
            entire_matrix /= np.linalg.norm(entire_matrix)

            entire_result = entire_matrix.dot(entire_matrix.dot(entire_matrix.dot(entire_vector)))
        else:
            entire_vector = None
            entire_matrix = None
            entire_result = None

        entire_result = communicator.bcast(entire_result, root=0)

        parallel_vector.broadcast(entire_vector, root=0)
        matrix.broadcast(entire_matrix, root=0)


        for i in range(3):
            matrix.dot(parallel_vector)

        self.assertLess(np.linalg.norm(parallel_vector.fullData()-entire_result), 1e-10)

    def testDotNotSqaureSized(self):

        n_rows = 300
        n_columns = 700

        parallel_vector_in = createVector(rows=n_rows, columns=n_columns)
        parallel_vector_out = createVector(rows=n_rows, columns=n_rows)
        matrix = ParallelMatrix(parallel_vector_in.distributionPlan())

        communicator = matrix.distributionPlan().communicator()

        if communicator.Get_rank() == 0:
            entire_vector = np.array(np.random.random(n_columns), dtype=np.complex128)
            entire_vector /= np.linalg.norm(entire_vector)

            entire_matrix = np.random.random((n_rows, n_columns))
            entire_matrix /= np.linalg.norm(entire_matrix)

            entire_result = entire_matrix.dot(entire_vector)
        else:
            entire_vector = None
            entire_matrix = None
            entire_result = None

        entire_result = communicator.bcast(entire_result, root=0)

        parallel_vector_in.broadcast(entire_vector, root=0)
        matrix.broadcast(entire_matrix, root=0)


        matrix.dot(parallel_vector_in, parallel_vector_out)

        self.assertLess(np.linalg.norm(parallel_vector_out.fullData()-entire_result), 1e-10)


    def testDotComplex(self):
        n_rows = 700
        n_columns = n_rows

        parallel_vector = createVector(rows=n_rows, columns=n_columns)
        matrix = ParallelMatrix(parallel_vector.distributionPlan())

        communicator = matrix.distributionPlan().communicator()

        if communicator.Get_rank() == 0:
            entire_vector = np.array(np.random.random(n_columns), dtype=np.complex128)
            entire_vector /= np.linalg.norm(entire_vector)

            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
            entire_matrix /= np.linalg.norm(entire_matrix)

            entire_result = entire_matrix.dot(entire_matrix.dot(entire_matrix.dot(entire_vector)))
        else:
            entire_vector = None
            entire_matrix = None
            entire_result = None

        entire_result = communicator.bcast(entire_result, root=0)

        parallel_vector.broadcast(entire_vector, root=0)
        matrix.broadcast(entire_matrix, root=0)


        for i in range(3):
            matrix.dot(parallel_vector)

        self.assertLess(np.linalg.norm(parallel_vector.fullData()-entire_result), 1e-10)

    def testDotComplexNotSquared(self):
        n_rows = 300
        n_columns = 700

        parallel_vector_in = createVector(rows=n_rows, columns=n_columns)
        parallel_vector_out = createVector(rows=n_rows, columns=n_rows)
        matrix = ParallelMatrix(parallel_vector_in.distributionPlan())


        matrix = ParallelMatrix(parallel_vector_in.distributionPlan())

        communicator = matrix.distributionPlan().communicator()

        if communicator.Get_rank() == 0:
            entire_vector = np.array(np.random.random(n_columns), dtype=np.complex128)
            entire_vector /= np.linalg.norm(entire_vector)

            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
            entire_matrix /= np.linalg.norm(entire_matrix)

            entire_result = entire_matrix.dot(entire_vector)
        else:
            entire_vector = None
            entire_matrix = None
            entire_result = None

        entire_result = communicator.bcast(entire_result, root=0)

        parallel_vector_in.broadcast(entire_vector, root=0)
        matrix.broadcast(entire_matrix, root=0)


        matrix.dot(parallel_vector_in, parallel_vector_out)

        self.assertLess(np.linalg.norm(parallel_vector_out.fullData()-entire_result), 1e-10)

    def testDotComplexConjugate(self):
        n_rows = 700
        n_columns = n_rows

        parallel_vector = createVector(rows=n_rows, columns=n_columns)
        matrix = ParallelMatrix(parallel_vector.distributionPlan())

        communicator = matrix.distributionPlan().communicator()

        if communicator.Get_rank() == 0:
            entire_vector = np.array(np.random.random(n_columns), dtype=np.complex128)
            entire_vector /= np.linalg.norm(entire_vector)

            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
            entire_matrix /= np.linalg.norm(entire_matrix)

            c_entire_matrix = entire_matrix.conjugate()
            entire_result = c_entire_matrix.dot(c_entire_matrix.dot(c_entire_matrix.dot(entire_vector)))
        else:
            entire_vector = None
            entire_matrix = None
            entire_result = None

        entire_result = communicator.bcast(entire_result, root=0)

        parallel_vector.broadcast(entire_vector, root=0)
        matrix.broadcast(entire_matrix, root=0)


        for i in range(3):
            matrix.dot(parallel_vector, complex_conjugate=True)

        self.assertLess(np.linalg.norm(parallel_vector.fullData()-entire_result), 1e-10)

    def testLocalMatrix(self):
        n_rows = 20
        n_columns = n_rows

        matrix = createMatrix(n_rows, n_columns)
        entire_matrix = np.random.random((n_rows, n_columns))
        matrix._local_matrix = entire_matrix.copy()

        self.assertLess(np.linalg.norm(entire_matrix-matrix.localMatrix()), 1e-10)

    def testAddition(self):
        n_rows = 300
        n_columns = n_rows

        parallel_vector = createVector(rows=n_rows, columns=n_columns)
        matrix = [ParallelMatrix(parallel_vector.distributionPlan()) for i in range(5)]

        communicator = matrix[0].distributionPlan().communicator()

        total_matrix = None
        for i in range(len(matrix)):
            if communicator.Get_rank() == 0:
                entire_matrix = np.random.random((n_rows, n_columns))
                entire_matrix /= np.linalg.norm(entire_matrix)

                if total_matrix is None:
                    total_matrix = entire_matrix
                else:
                    total_matrix += entire_matrix
            else:
                entire_matrix = None

            matrix[i].broadcast(entire_matrix, root=0)

        total_matrix = communicator.bcast(total_matrix, root=0)

        for i in range(1, len(matrix)):
            matrix[0] += matrix[i]

        for i_local_row, i_row in enumerate(matrix[0].localRows()):
            self.assertLess(np.linalg.norm(matrix[0].localMatrix()[i_local_row, :] - total_matrix[i_row, :]), 1e-10)

    def testScalarMultiplication(self):
        n_rows = 300
        n_columns = n_rows

        parallel_vector = createVector(rows=n_rows, columns=n_columns)
        matrix = ParallelMatrix(parallel_vector.distributionPlan())

        scalars = np.random.random(5)

        communicator = matrix.distributionPlan().communicator()

        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns))
            entire_matrix /= np.linalg.norm(entire_matrix)
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)
        entire_matrix = communicator.bcast(entire_matrix, root=0)

        for scalar in scalars:
            matrix = scalar * matrix
            matrix = matrix * scalar
            matrix *= scalar

            entire_matrix = scalar * entire_matrix
            entire_matrix = entire_matrix * scalar
            entire_matrix *= scalar

            for i_local_row, i_row in enumerate(matrix.localRows()):
                self.assertLess(np.linalg.norm(matrix.localMatrix()[i_local_row, :] - entire_matrix[i_row, :]), 1e-10)

    def testTrace(self):
        n_rows = 3000
        n_columns = n_rows

        matrix = createMatrix(rows=n_rows, columns=n_columns)

        communicator = matrix.distributionPlan().communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        trace = matrix.trace()

        if communicator.Get_rank() == 0:
            entire_trace = np.array([entire_matrix[i, i] for i in range(n_rows)])
            self.assertLess(np.linalg.norm(trace-entire_trace), 1e-12)

    def testGatherMatrix(self):
        n_rows = 300
        n_columns = n_rows

        matrix = createMatrix(rows=n_rows, columns=n_columns)

        communicator = matrix.distributionPlan().communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        gathered_matrix = matrix.gatherMatrix(root=0)

        if communicator.Get_rank() == 0:
            self.assertLess(np.linalg.norm(gathered_matrix-entire_matrix), 1e-12)

    def testTransposition(self):
        n_rows = 302
        n_columns = n_rows

        matrix = createMatrix(rows=n_rows, columns=n_columns)

        communicator = matrix.distributionPlan().communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        transposed = matrix.transpose()
        gathered_matrix = transposed.gatherMatrix(root=0)

        if communicator.Get_rank() == 0:
            self.assertLess(np.linalg.norm(gathered_matrix-entire_matrix.transpose()), 1e-12)

    def testEnlargeTo(self):
        n_rows = 130
        n_columns = n_rows

        n_rows_larger = 600
        n_columns_larger = n_rows_larger

        matrix = createMatrix(rows=n_rows, columns=n_columns)
        new_matrix = createMatrix(rows=n_rows_larger, columns=n_columns_larger)


        communicator = matrix.distributionPlan().communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        new_matrix = matrix.enlargeTo(new_matrix.distributionPlan())
        gathered_matrix = new_matrix.gatherMatrix(root=0)

        if communicator.Get_rank() == 0:
            gathered_matrix[0:n_rows, 0:n_columns] -= entire_matrix[:,:]
            self.assertLess(np.linalg.norm(gathered_matrix), 1e-12)

    def testShrinkTo(self):
        n_rows = 6
        n_columns = n_rows

        n_rows_shrink = 4
        n_columns_shrink = 3

        matrix = createMatrix(rows=n_rows, columns=n_columns)
        new_matrix = createMatrix(rows=n_rows_shrink, columns=n_columns_shrink)


        communicator = matrix.distributionPlan().communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        new_matrix = matrix.shrinkTo(new_matrix.distributionPlan())
        gathered_matrix = new_matrix.gatherMatrix(root=0)

        if communicator.Get_rank() == 0:
            entire_matrix[0:n_rows_shrink, 0:n_columns_shrink] -= gathered_matrix[:,:]
            self.assertLess(np.linalg.norm(entire_matrix[0:n_rows_shrink, 0:n_columns_shrink]), 1e-12)

    def testDotForTransposed(self):

        n_rows = 700
        n_columns = n_rows

        parallel_vector = createVector(rows=n_rows, columns=n_columns)
        matrix = ParallelMatrix(parallel_vector.distributionPlan())

        communicator = matrix.distributionPlan().communicator()

        if communicator.Get_rank() == 0:
            entire_vector = np.array(np.random.random(n_columns), dtype=np.complex128)
            entire_vector /= np.linalg.norm(entire_vector)

            entire_matrix = np.random.random((n_rows, n_columns))
            entire_matrix /= np.linalg.norm(entire_matrix)

            entire_result = entire_matrix.transpose().dot(entire_vector)
        else:
            entire_vector = None
            entire_matrix = None
            entire_result = None

        entire_result = communicator.bcast(entire_result, root=0)

        parallel_vector.broadcast(entire_vector, root=0)
        matrix.broadcast(entire_matrix, root=0)
        matrix.dotForTransposed(parallel_vector)
        self.assertLess(np.linalg.norm(parallel_vector.fullData()-entire_result), 1e-10)

    def testDotForTransposedNotSquared(self):

        n_rows = 300
        n_columns = 700

        parallel_vector_in = createVector(rows=n_rows, columns=n_rows)
        parallel_vector_out = createVector(rows=n_columns, columns=n_columns)
        matrix = createMatrix(n_rows, n_columns)

        communicator = matrix.distributionPlan().communicator()

        if communicator.Get_rank() == 0:
            entire_vector = np.array(np.random.random(n_rows), dtype=np.complex128)
            entire_vector /= np.linalg.norm(entire_vector)

            entire_matrix = np.random.random((n_rows, n_columns))
            entire_matrix /= np.linalg.norm(entire_matrix)

            entire_result = entire_matrix.transpose().dot(entire_vector)
        else:
            entire_vector = None
            entire_matrix = None
            entire_result = None

        entire_result = communicator.bcast(entire_result, root=0)

        parallel_vector_in.broadcast(entire_vector, root=0)
        matrix.broadcast(entire_matrix, root=0)
        matrix.dotForTransposed(parallel_vector_in, parallel_vector_out)
        self.assertLess(np.linalg.norm(parallel_vector_out.fullData()-entire_result), 1e-10)

    def testGlobalRow(self):
        n_rows = 230
        n_columns = 800

        matrix = createMatrix(rows=n_rows, columns=n_columns)

        communicator = matrix.distributionPlan().communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        for i_rank in matrix.distributionPlan().ranks():
            result = matrix.gatherMatrix(root=i_rank)

            if i_rank == matrix.distributionPlan().myRank():
                gathered_matrix = result


        for i_row in range(n_rows):
            row = matrix.globalRow(i_row)
            self.assertLess(np.linalg.norm(gathered_matrix[i_row,:]-row), 1e-12)

    def testCachedGlobalRow(self):
        n_rows = 230
        n_columns = 800

        matrix = createMatrix(rows=n_rows, columns=n_columns)

        communicator = matrix.distributionPlan().communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        for i_rank in matrix.distributionPlan().ranks():
            result = matrix.gatherMatrix(root=i_rank)

            if i_rank == matrix.distributionPlan().myRank():
                gathered_matrix = result


        for i_row in range(n_rows):
            row = matrix.cachedGlobalRow(i_row)
            row_uncached = matrix.globalRow(i_row)

            self.assertLess(np.linalg.norm(row-row_uncached), 1e-12)
            self.assertLess(np.linalg.norm(gathered_matrix[i_row,:]-row), 1e-12)
