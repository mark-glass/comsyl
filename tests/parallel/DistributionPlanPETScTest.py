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
from comsyl.parallel.DistributionPlanPETSc import DistributionPlanPETSc

import sys
from mpi4py import MPI as mpi
from petsc4py import PETSc

import numpy

def createPETScMatrix(n_rows, n_columns):
    opts = PETSc.Options()
    n = opts.getInt('n', 1000)

    A = PETSc.Mat().create()
    A.setSizes([n_rows, n_columns])
    A.setFromOptions()
    A.setUp()

    return A

class DistributionPlanPETScTest(unittest.TestCase):
    def testConstructor(self):

        n_rows = 50
        n_columns = 100

        petsc_matrix = createPETScMatrix(n_rows, n_columns)
        plan = DistributionPlanPETSc(mpi.COMM_WORLD, petsc_matrix)

        blocks = petsc_matrix.getOwnershipRanges()

        for i_row in range(n_rows):
            i_owner = plan.rankByGlobalIndex(i_row)
            self.assertTrue(blocks[i_owner] <= i_row < blocks[i_owner+1])

        my_range = petsc_matrix.getOwnershipRange()
        for i_row in range(my_range[0], my_range[1]):
            self.assertTrue(i_row in plan.localRows())

        for i_row in plan.localRows():
            self.assertTrue(my_range[0] <= i_row < my_range[1])