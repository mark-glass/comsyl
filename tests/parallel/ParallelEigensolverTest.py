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

from comsyl.parallel.ParallelMatrix import ParallelMatrix
from comsyl.parallel.ParallelEigensolver import ParallelEigensolver
from comsyl.parallel.ParallelLinearOperator import ParallelLinearOperator
from tests.parallel.ParallelVectorTest import createVector
from tests.parallel.ParallelMatrixTest import createMatrix
from comsyl.math.Eigensolver import Operator, Eigensolver

def parallelDiagonalization():
    n_rows = 100
    n_columns = n_rows
    matrix = createMatrix(rows=n_rows, columns=n_columns)

    communicator = matrix.communicator()
    if communicator.Get_rank() == 0:
        entire_matrix = np.array(np.diag(np.arange(n_rows)), dtype=np.complex128)
    else:
        entire_matrix = None

    matrix.broadcast(entire_matrix, root=0)

    operator = ParallelLinearOperator(matrix)

    operator.listenIfSlave()

    if communicator.Get_rank() == 0:
        #S=entire_matrix
        #S2n = np.hstack((np.vstack((S.real,-S.imag)), np.vstack((S.imag,S.real)))).real
        #eigenvalue, eigenfunctions = eigs(entire_matrix, 190)
        op = Operator.fromLinearOperator(operator)
        solver = Eigensolver()

        number_eigenvectors = 10
        eigenvalue, eigenfunctions = solver.arnoldi(op, number_eigenvectors)
        print(eigenvalue)

    operator.finishListen()



class ParallelEigensolverTest(unittest.TestCase):

    def testArnoldi(self):
        n_rows = 500
        n_columns = n_rows
        matrix = createMatrix(rows=n_rows, columns=n_columns)

        communicator = matrix.communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.array(np.diag(np.arange(n_rows)), dtype=np.complex128)
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        solver = ParallelEigensolver()
        number_eigenvectors = 50
        eigenvalue, eigenfunctions = solver.arnoldi(matrix, n = number_eigenvectors)

        #print(eigenvalue)
        #print(eigenfunctions)
        #parallelDiagonalization()


        self.assertLess(np.linalg.norm(eigenvalue - np.arange(n_rows-number_eigenvectors, n_rows)[::-1]), 1e-8)

        for i in range(eigenfunctions.shape[1]):
            self.assertLess(np.abs(np.linalg.norm(eigenfunctions[:, i])-1), 1e-8)
            self.assertEqual(np.abs(eigenfunctions[:,i]).argmax(), n_rows-1-i)

    @unittest.skip
    def testParallelDiagonalizationComplex(self):
        n_rows = 10
        n_columns = n_rows
        matrix = createMatrix(rows=n_rows, columns=n_columns)


        communicator = matrix.communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.array(np.random.random((n_rows,n_columns)), dtype=np.complex128)
            entire_matrix += 1j* np.random.random((n_rows,n_columns))
            entire_matrix = entire_matrix.conj().transpose()+entire_matrix
        else:
            entire_matrix = None

        matrix.broadcast(entire_matrix, root=0)

        operator = ParallelLinearOperator(matrix)

        operator.listenIfSlave()

        if communicator.Get_rank() == 0:
            op  = Operator.fromLinearOperator(operator)
            eig = Eigensolver()

            number_eigenfunctions = 100
            eigenvalue, eigenfunctions = eig.arnoldi(op, number_eigenfunctions, 1e-13)
            eigenvalue_c, eigenfunctions_c = np.linalg.eigh(entire_matrix)

            s_eigenvalue_c = eigenvalue_c[eigenvalue_c>0.0]
            s_eigenfunctions_c = eigenfunctions_c[:, eigenvalue_c>0.0]
            s_eigenvalue_c = s_eigenvalue_c[s_eigenvalue_c.argsort()[::-1]]
            s_eigenfunctions_c = s_eigenfunctions_c[:,s_eigenvalue_c.argsort()]

            def rayleigh(matrix, vector):
                return vector.conj().dot(matrix.dot(vector))/vector.conj().dot(vector)

            for i in range(number_eigenfunctions):
                error_rayleigh = np.abs(rayleigh(entire_matrix, eigenfunctions[:, i])-s_eigenvalue_c[i])
                self.assertLess(error_rayleigh, 1e-10)

                error_ritz = np.linalg.norm(entire_matrix.dot(eigenfunctions[:, i]) - eigenvalue[i] * eigenfunctions[:,i])
                self.assertLess(error_ritz, 1e-10)

                alpha =  eigenfunctions[0, i] / s_eigenfunctions_c[0, i]
                error_l2 = np.linalg.norm(eigenfunctions[:, i] - alpha * s_eigenfunctions_c[:, i])
                self.assertLess(error_l2, 1e-10)

        operator.finishListen()