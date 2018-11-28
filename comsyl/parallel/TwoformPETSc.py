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
from comsyl.mathcomsyl.Eigenmoder import Eigenmoder
from comsyl.parallel.ParallelMatrixPETSc import ParallelMatrixPETSc
from comsyl.parallel.ParallelVector import ParallelVector
from comsyl.parallel.DistributionPlan import DistributionPlan
import mpi4py.MPI as mpi

class TwoformPETSc(object):
    def __init__(self, twoform):
        self._parent = twoform

        vector = self._parent.vector(0)
        self._n_size = vector.size
        self._petsc_matrix = PETSc.Mat().create()
        self._petsc_matrix.setSizes([self._n_size, self._n_size])
        self._petsc_matrix.setUp()

        self._parallel_matrix = ParallelMatrixPETSc(self._petsc_matrix)
        plan = self._parallel_matrix.distributionPlan()
        self._parent._distribution_plan = plan
        self._vector_in = ParallelVector(plan)
        self._vector_out = ParallelVector(plan)
        self._distribution_plan = DistributionPlan(communicator=mpi.COMM_WORLD, n_columns=self.dimensionSize(), n_rows=self.dimensionSize())

    def mult(self, A, x, y):
        xx = x.getArray(readonly=1)
        yy = y.getArray(readonly=0)

        yy[:] =  self._1d_dot(xx)

    def _1d_dot(self, xx):
        x_2d = self.from1dTo2d(xx)
        result_2d = self._parent.dot(x_2d)
        return self.from2dTo1d(result_2d)

    def from1dTo2d(self, x_1d):
        x_2d = x_1d.reshape((len(self.xCoordinates()),
                             len(self.yCoordinates())))

        return x_2d

    def from2dTo1d(self, x_2d):
        x_1d = x_2d.reshape(len(self.xCoordinates()) * len(self.yCoordinates()))

        return x_1d

    def xCoordinates(self):
        return self._parent.xCoordinates()

    def yCoordinates(self):
        return self._parent.yCoordinates()

    def dimensionSize(self):
        return self._n_size

    def totalShape(self):
        return (self._n_size, self._n_size)

    def trace(self):
        return self._parent.intensity()

    def petScMatrix(self):
        context = self
        A = PETSc.Mat().createPython([self.dimensionSize(),self.dimensionSize()], context)
        A.setUp()
        return A
    def communicator(self):
        return self._distribution_plan.communicator()

    def distributionPlan(self):
        return self._distribution_plan

    def dot(self, v_in, v_out=None):
        if v_out is None:
            v_out = v_in

        v_out.broadcast(self._1d_dot(v_in.fullData()), root=0)

    def releaseMemory(self):
        pass

    def getVecs(self):
        return self._petsc_matrix.getVecs()

    def diagonalize(self, target_number_modes=50):
        print("Doing diagonalization")
        new_twoform = Eigenmoder(self.xCoordinates(), self.yCoordinates()).eigenmodes(self, target_number_modes)
        new_twoform._eigenvalues /= np.diff(self.xCoordinates())[0] * np.diff(self.yCoordinates())[0]
        return new_twoform
