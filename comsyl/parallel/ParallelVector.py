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
import mpi4py.MPI as mpi

class ParallelVector(object):
    def __init__(self, distribution_plan, type="collective"):
        self._distribution_plan = distribution_plan
        self._local_data = np.zeros(self.distributionPlan().localShape()[0], dtype=np.complex128)

        if type=="collective":
            self._full_data = np.zeros(self.distributionPlan().totalShape()[0], dtype=np.complex128)
        else:
            raise NotImplementedError

    def clone(self):
        cloned_vector = ParallelVector(self.distributionPlan())
        cloned_vector.setLocal(self._local_data.copy())
        cloned_vector._full_data = self._full_data.copy()

        return cloned_vector

    def distributionPlan(self):
        return self._distribution_plan

    def communicator(self):
        return self.distributionPlan().communicator()

    def localColumns(self):
        return self.distributionPlan().localColumns()

    def setLocal(self, local_data):
        self._local_data[:] = local_data[:]

    def localVector(self):
        return self._local_data

    def broadcast(self, data=None, root=0):
        self._full_data = self.communicator().bcast(data, root)

        local_min = 0
        local_max = len(self.distributionPlan().localRows()) - 1
        global_min = self.distributionPlan().localToGlobalIndex(local_min)
        global_max = self.distributionPlan().localToGlobalIndex(local_max)

        self._local_data[:] = self._full_data[global_min:global_max+1]

    def setCollective(self, local_data):
        self.setLocal(local_data)

        local_min = 0
        local_max = len(self.distributionPlan().localRows()) - 1
        global_min = self.distributionPlan().localToGlobalIndex(local_min)
        global_max = self.distributionPlan().localToGlobalIndex(local_max)

        self._full_data[:] = 0.0
        self._full_data[global_min:global_max+1] = local_data[:]

        sendbuf = self._full_data.copy()
        self.communicator().Allreduce(sendbuf, self._full_data, op=mpi.SUM)

    def sumFullData(self, full_data):
        self._full_data[:] = full_data[:]

        sendbuf = self._full_data.copy()
        self.communicator().Allreduce(sendbuf, self._full_data, op=mpi.SUM)

        local_min = 0
        local_max = len(self.distributionPlan().localRows()) - 1
        global_min = self.distributionPlan().localToGlobalIndex(local_min)
        global_max = self.distributionPlan().localToGlobalIndex(local_max)

        self.setLocal(self._full_data[global_min:global_max+1])

    def fullData(self):
        return self._full_data