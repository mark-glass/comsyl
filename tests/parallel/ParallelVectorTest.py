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
from comsyl.parallel.ParallelVector import ParallelVector
from mpi4py import MPI as mpi

def createDistributionPlan(rows=10, columns=10):
    return DistributionPlan(mpi.COMM_WORLD, n_rows=rows, n_columns=columns)

def createVector(rows=10, columns=10):
    plan = createDistributionPlan(rows, columns)
    vector = ParallelVector(plan)

    return vector


class ParallelVectorTest(unittest.TestCase):
    def testLocalColumns(self):
        n_rows = 10
        vector = createVector(rows=n_rows)

        rows = vector.localColumns()

        self.assertLess(np.linalg.norm(rows-np.arange(n_rows)), 1e-10)


    def testSetLocal(self):
        n_rows = 10
        vector = createVector(rows=n_rows)

        n_rows_local = vector.distributionPlan().localShape()[0]
        data = np.array(np.random.random(n_rows_local), dtype=np.complex128)

        vector.setLocal(data)

        self.assertLess(np.linalg.norm(vector.localVector() - data), 1e-10)

    def testLocalVector(self):
        n_rows = 100
        vector = createVector(rows=n_rows)

        n_rows_local = vector.distributionPlan().localShape()[0]
        data = np.array(np.random.random(n_rows_local), dtype=np.complex128)

        vector.setLocal(data)

        self.assertLess(np.linalg.norm(vector.localVector() - data), 1e-10)

    def testSetCollective(self):
        n_rows = 10
        vector = createVector(rows=n_rows)

        n_rows_local = vector.distributionPlan().localShape()[0]
        data = np.array(np.random.random(n_rows_local), dtype=np.complex128)

        vector.setCollective(data)

        global_min = vector.distributionPlan().localToGlobalIndex(0)
        global_max = vector.distributionPlan().localToGlobalIndex(n_rows_local-1)

        self.assertLess(np.linalg.norm(vector.fullData()[global_min:global_max+1] - data), 1e-10)

    def testSumFullData(self):
        n_rows = 10
        vector = createVector(rows=n_rows)

        local_data = np.array(np.random.random(n_rows), dtype=np.complex128)

        vector.sumFullData(local_data)

        sendbuf = local_data.copy()
        sum_data = np.zeros_like(sendbuf)
        vector.communicator().Allreduce(sendbuf, sum_data, op=mpi.SUM)

        self.assertLess(np.linalg.norm(vector.fullData() - sum_data), 1e-10)

    def testClone(self):
        n_rows = 300
        vector = createVector(rows=n_rows)

        n_rows_local = vector.distributionPlan().localShape()[0]
        data = np.array(np.random.random(n_rows_local), dtype=np.complex128)

        vector.setCollective(data)
        cloned_vector = vector.clone()

        self.assertLess(np.linalg.norm(vector._local_data - cloned_vector._local_data), 1e-12)
        self.assertLess(np.linalg.norm(vector.fullData() - cloned_vector.fullData()), 1e-12)


    def testFullData(self):
        return