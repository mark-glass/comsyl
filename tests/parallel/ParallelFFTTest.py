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

from comsyl.parallel.ParallelFFT import ParallelFFT
from comsyl.parallel.ParallelFunction2D import ParallelFunction2D
from mpi4py import MPI as mpi

from tests.parallel.ParallelMatrixTest import createMatrix


class ParallelFFTTest(unittest.TestCase):
    def testFFT(self):
        n_rows = 3002
        n_columns = n_rows

        matrix = createMatrix(rows=n_rows, columns=n_columns)

        communicator = matrix.distributionPlan().communicator()
        if communicator.Get_rank() == 0:
            entire_matrix = np.random.random((n_rows, n_columns)) + 1j*np.random.random((n_rows, n_columns))
            local_fft = np.fft.fft2(entire_matrix)
        else:
            entire_matrix = None


        matrix.broadcast(entire_matrix, root=0)
        func = ParallelFunction2D.fromParallelMatrix(matrix)
        pfft = ParallelFFT()
        pfft.fft(func)

        gathered_matrix = func._matrix.gatherMatrix(root=0)

        if communicator.Get_rank() == 0:
            self.assertLess(np.linalg.norm(gathered_matrix-local_fft), 1e-12)