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

from comsyl.parallel.ParallelMatrix import ParallelMatrix
from comsyl.parallel.DistributionPlanPETSc import DistributionPlanPETSc
    #
    # def __getitem__(self, ij):
    #     """Similar to `Mat.getValues()`"""
    #     i, j = ij
    #     if isinstance(i, int) and isinstance(j, int):
    #         return self.getValue(i, j)
    #     msize  = _petsc.MatGetSize(self)
    #     arange = _numpy.arange
    #     dtype  = _petsc.PetscInt
    #     if isinstance(i, slice):
    #         start, stop, stride = i.indices(msize[0])
    #         i = arange(start, stop, stride, dtype)
    #     if isinstance(j, slice):
    #         start, stop, stride = j.indices(msize[1])
    #         j = arange(start, stop, stride, dtype)
    #     return self.getValues(i, j)
    #
    # def __setitem__(self, ij, v):
    #     """Similar to `Mat.setValues()`"""
    #     i, j = ij
    #     msize  = _petsc.MatGetSize(self)
    #     arange = _numpy.arange
    #     dtype  = _petsc.PetscInt
    #     if isinstance(i, slice):
    #         start, stop, stride = i.indices(msize[0])
    #         i = arange(start, stop, stride, dtype)
    #     if isinstance(j, slice):
    #         start, stop, stride = j.indices(msize[1])
    #         j = arange(start, stop, stride, dtype)
    #     self.setValues(i, j, v)
    #
    # def __call__(self, x, y=None):
    #     """Similar to as `Mat.mult()`"""
    #     if y is None:
    #         y = _petsc.MatGetVecLeft(self)
    #     _petsc.MatMult(self, x, y)
    #     return y

class ParallelMatrixPETSc(ParallelMatrix):
    def __init__(self, petsc_matrix):
        self._petsc_matrix = petsc_matrix
        self._distribution_plan = DistributionPlanPETSc(communicator=mpi.COMM_WORLD, petsc_object=petsc_matrix)
        self._column_indices = np.arange(self.distributionPlan()._n_columns, dtype=np.int64)

    def petScMatrix(self):
        return self._petsc_matrix

    def setElement(self, global_index_row, global_index_column, content):
        self._petsc_matrix[global_index_row, global_index_column] = content

    def setRow(self, global_index, content):
        self._petsc_matrix[global_index, :] = content
        #self._petsc_matrix.setValues(global_index, self._column_indices, content)

    def dot(self, parallel_vector_in, parallel_vector_out=None, complex_conjugate=False):

        if complex_conjugate:
            raise NotImplementedError

        vec_in, tmp = self.petScMatrix().getVecs()
        vec_out, tmp = self.petScMatrix().getVecs()

        i_start = self.localRows().min()
        i_end = self.localRows().max()
        vec_in[i_start:i_end+1] = parallel_vector_in.localVector()

        self.petScMatrix().mult(vec_in, vec_out)

        if parallel_vector_out is None:
            parallel_vector_out = parallel_vector_in

        parallel_vector_out.setCollective(vec_out[i_start:i_end+1])

    def dotForTransposed(self, parallel_vector_in, parallel_vector_out=None):
        raise NotImplementedError

    def localMatrix(self):
        return self._local_matrix

    def gatherMatrix(self, root=0):
        raise NotImplementedError

    def broadcast(self, entire_matrix, root=0):
        raise NotImplementedError

    def globalRow(self, global_row_index):
        return self._petsc_matrix[global_row_index,:]

    def __add__(self, other):
        self._testEqualDistribution(other.distributionPlan())

        self._petsc_matrix += other._petsc_matrix
        return self

    def __mul__(self, scalar):
        self._petsc_matrix *= scalar
        return self

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def trace(self, root=0):
        my_trace = []
        for i_local, i_global in enumerate(self.localRows()):
            my_trace.append(self._petsc_matrix[i_global, i_global])

        my_trace = np.array(my_trace)

        total_trace = np.array(self.communicator().allgather(my_trace))
        total_trace = np.hstack(total_trace)
        return total_trace

    def transpose(self):
        self._petsc_matrix.transpose()

    def localRow(self, local_index):
        global_index = self.distributionPlan().localToGlobalIndex(local_index)
        return self.globalRow(global_index)

    def setLocalRow(self, local_index, row):
        global_index = self.distributionPlan().localToGlobalIndex(local_index)
        self.setRow(global_index, row)

    def enlargeTo(self, new_distribution_plan):
        raise NotImplementedError

    def shrinkTo(self, new_distribution_plan):
        raise NotImplementedError

    def assemble(self):
        self.petScMatrix().assemble()

    def releaseMemory(self):
        self._petsc_matrix.destroy()