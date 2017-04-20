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

class ParallelMatrix(object):
    def __init__(self, distribution_plan):
        self._distribution_plan = distribution_plan

        self._local_matrix = np.zeros(
                                      self.localShape(),
                                      dtype=np.complex128,
                                     )

        self._cache_global_row_index_start = None
        self._cache_global_row_index_end = None

    def communicator(self):
        return self._distribution_plan.communicator()

    def distributionPlan(self):
        return self._distribution_plan

    def _testEqualDistribution(self, distribution_plan):
        if distribution_plan != self._distribution_plan:
            raise Exception("Can only multiply equally distributed objects.")

    def localShape(self):
        return self._distribution_plan.localShape()

    def localRows(self):
        return self._distribution_plan.localRows()

    def localColumns(self):
        return self._distribution_plan.localColumns()

    def totalShape(self):
        totalShape = self.distributionPlan().totalShape()
        return totalShape

    def setElement(self, global_index_row, global_index_column, content):
        local_index = self._distribution_plan.globalToLocalIndex(global_index_row)
        self._local_matrix[local_index, global_index_column] = content

    def setRow(self, global_index, content):
        local_index = self._distribution_plan.globalToLocalIndex(global_index)
        self._local_matrix[local_index, :] = content

    def dot(self, parallel_vector_in, parallel_vector_out=None, complex_conjugate=False):
#        self._testEqualDistribution(parallel_vector_in.distributionPlan())

        matrix = self.localMatrix()
        if complex_conjugate:
            matrix = matrix.conjugate()

        dot_result = matrix.dot(parallel_vector_in.fullData())

        if parallel_vector_out is None:
            parallel_vector_out = parallel_vector_in

        parallel_vector_out.setCollective(dot_result)

    def dotForTransposed(self, parallel_vector_in, parallel_vector_out=None):
        if parallel_vector_out is None:
            parallel_vector_out = parallel_vector_in

# TODO Test inversion
#        self._testEqualDistribution(parallel_vector_out.distributionPlan())

        start_row = self.localRows().min()
        end_row = self.localRows().max()

        transposed_vector = parallel_vector_in.fullData()[start_row:end_row+1].transpose()

        dot_result = transposed_vector.dot(self.localMatrix())

        parallel_vector_out.sumFullData(dot_result)

    def localMatrix(self):
        return self._local_matrix

    def gatherMatrix(self, root=0):
        #TODO implement faster with .Gather, .gather is 2gb limited.

        gathered_matrix = np.zeros(self.totalShape(),dtype=self.localMatrix().dtype)

        my_rank = self.distributionPlan().myRank()
        communicator = self.communicator()
        blocks = self.distributionPlan().rowDistribution()

        for i_sender in self.distributionPlan().ranks():

            if my_rank==root:
                block = blocks[i_sender]
                i_start_index = block[0]
                i_end_index = block[1]

                if i_sender==root:
                    gathered_matrix[i_start_index:i_end_index+1, :] = self.localMatrix()
                else:
                    gathered_matrix[i_start_index:i_end_index+1, :] = communicator.recv(source=i_sender, tag=13)

            else:
                if my_rank==i_sender:
                    communicator.send(self.localMatrix(), dest=root, tag=13)

        return gathered_matrix

    def broadcast(self, entire_matrix, root=0):
        entire_matrix = self.communicator().bcast(entire_matrix, root)

        for i_local_rows in self.localRows():
            global_index = i_local_rows
            self.setRow(global_index, entire_matrix[global_index, :])

    def globalRow(self, global_row_index):
        sender = self.distributionPlan().rankByGlobalIndex(global_row_index)
        distribution_plan = self.distributionPlan()

        if distribution_plan.myRank() == sender:
            local_row_index = distribution_plan.globalToLocalIndex(global_row_index)
            row = self.localMatrix()[local_row_index, :]
        else:
            row = self.localMatrix()[0, :].copy()

        row = self.communicator().bcast(row, sender)
        return row

    def cachedGlobalRow(self, global_row_index):

        if self._cache_global_row_index_start is None or\
           global_row_index < self._cache_global_row_index_start or\
           self._cache_global_row_index_end < global_row_index:

            sender = self.distributionPlan().rankByGlobalIndex(global_row_index)
            distribution_plan = self.distributionPlan()

            rows_indices = distribution_plan.rows(sender)
            self._cache_global_row_index_start = rows_indices.min()
            self._cache_global_row_index_end = rows_indices.max()

            if distribution_plan.myRank() == sender:
                matrix = self.localMatrix()[:, :]
            else:
                matrix = np.zeros((rows_indices.size, self.localMatrix().shape[1]), dtype=self.localMatrix().dtype)

            self._cached_rows = self.communicator().bcast(matrix, sender)

        row_index = global_row_index - self._cache_global_row_index_start

        return self._cached_rows[row_index, :]

    def resetCacheGlobalRow(self):
        self._cache_global_row_index_start = None
        self._cache_global_row_index_end = None
        self._cached_rows = None

    def __add__(self, other):
        self._testEqualDistribution(other.distributionPlan())

        self._local_matrix += other.localMatrix()
        return self

    def __mul__(self, scalar):
        self._local_matrix *= scalar
        return self

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def trace(self, root=0):
        my_trace = []
        for i_local, i_global in enumerate(self.localRows()):
            my_trace.append(self.localMatrix()[i_local, i_global])

        my_trace = np.array(my_trace)

        total_trace = np.array(self.communicator().allgather(my_trace))
        total_trace = np.hstack(total_trace)
        return total_trace

    def transpose(self):

        communicator = self.communicator()
        transposed = ParallelMatrix(self.distributionPlan())

        for i_active_root in self.distributionPlan().ranks():

            blocks = self.distributionPlan().rowDistribution()

            for i_rank, block in enumerate(blocks):
                if i_active_root == self.distributionPlan().myRank():
                    data_block = self.localMatrix()[:, block[0]:block[1]+1]
                    data_block = data_block.transpose()

                    if i_active_root != i_rank:
                        communicator.send(data_block, dest=i_rank, tag=11)
                    else:
                        transposed.localMatrix()[:, block[0]:block[1]+1] = data_block[:, :]

                elif i_rank == self.distributionPlan().myRank():
                    t_block = blocks[i_active_root]
                    transposed.localMatrix()[:, t_block[0]:t_block[1]+1] = communicator.recv(source=i_active_root, tag=11)

        return transposed

    def localRow(self, local_index):
        return self._local_matrix[local_index, :]

    def setLocalRow(self, local_index, row):
        self._local_matrix[local_index, :] = row

    def _transferBlock(self, i_sender, i_receiver, i_global_start_index, i_global_end_index, new_matrix):
        communicator = self.distributionPlan().communicator()
        my_rank = self.distributionPlan().myRank()
        current_size = self.totalShape()[1]

        if my_rank == i_sender and my_rank == i_receiver:
            i_start_index = self.distributionPlan().globalToLocalIndex(i_global_start_index)
            i_end_index = self.distributionPlan().globalToLocalIndex(i_global_end_index)

            i_new_start_index = new_matrix.distributionPlan().globalToLocalIndex(i_global_start_index)
            i_new_end_index = new_matrix.distributionPlan().globalToLocalIndex(i_global_end_index)

            new_matrix.localMatrix()[i_new_start_index:i_new_end_index+1, 0:current_size] = self.localMatrix()[i_start_index:i_end_index+1,:]
        elif my_rank == i_sender:
            i_start_index = self.distributionPlan().globalToLocalIndex(i_global_start_index)
            i_end_index = self.distributionPlan().globalToLocalIndex(i_global_end_index)

            communicator.send(self.localMatrix()[i_start_index:i_end_index+1, :], dest=i_receiver, tag=12)
        elif my_rank == i_receiver:
            i_new_start_index = new_matrix.distributionPlan().globalToLocalIndex(i_global_start_index)
            i_new_end_index = new_matrix.distributionPlan().globalToLocalIndex(i_global_end_index)

            new_matrix.localMatrix()[i_new_start_index:i_new_end_index+1, 0:current_size] = communicator.recv(source=i_sender, tag=12)

    def enlargeTo(self, new_distribution_plan):
        new_matrix = ParallelMatrix(new_distribution_plan)

        for i_sender, block in enumerate(self.distributionPlan().rowDistribution()):
            i_receiver = None
            for i_global_index in range(block[0], block[1]+1):
                target_rank = new_distribution_plan.rankByGlobalIndex(i_global_index)
                if i_receiver is None:
                    i_receiver = target_rank
                    i_start_index = i_global_index
                elif i_receiver != target_rank:
                    self._transferBlock(i_sender, i_receiver, i_start_index, i_global_index-1, new_matrix)
                    i_receiver = target_rank
                    i_start_index = i_global_index

            if i_receiver is not None:
                self._transferBlock(i_sender, i_receiver, i_start_index, block[1], new_matrix)

        return new_matrix

    def _transferBlockShrink(self, i_sender, i_receiver, i_global_start_index, i_global_end_index, new_matrix):

        communicator = self.distributionPlan().communicator()
        my_rank = self.distributionPlan().myRank()
        new_size = new_matrix.totalShape()[1]

        #print("rank/sender/recv/gstart/gend/nsize",my_rank,i_sender, i_receiver, i_global_start_index, i_global_end_index, new_size)

        if my_rank == i_sender and my_rank == i_receiver:
            i_start_index = self.distributionPlan().globalToLocalIndex(i_global_start_index)
            i_end_index = self.distributionPlan().globalToLocalIndex(i_global_end_index)

            i_new_start_index = new_matrix.distributionPlan().globalToLocalIndex(i_global_start_index)
            i_new_end_index = new_matrix.distributionPlan().globalToLocalIndex(i_global_end_index)

#            print("rank/start/end/nstart,nend",my_rank, i_start_index,i_end_index, i_new_start_index, i_new_end_index)
            new_matrix.localMatrix()[i_new_start_index:i_new_end_index+1, :] = self.localMatrix()[i_start_index:i_end_index+1, 0:new_size]
        elif my_rank == i_sender:
            i_start_index = self.distributionPlan().globalToLocalIndex(i_global_start_index)
            i_end_index = self.distributionPlan().globalToLocalIndex(i_global_end_index)

#            print("rank/start/end",my_rank, i_start_index,i_end_index)

            communicator.send(self.localMatrix()[i_start_index:i_end_index+1, 0:new_size], dest=i_receiver, tag=12)
        elif my_rank == i_receiver:
            i_new_start_index = new_matrix.distributionPlan().globalToLocalIndex(i_global_start_index)
            i_new_end_index = new_matrix.distributionPlan().globalToLocalIndex(i_global_end_index)

#            print("rank/nstart/nend",my_rank, i_new_start_index,i_new_end_index)

            new_matrix.localMatrix()[i_new_start_index:i_new_end_index+1, :] = communicator.recv(source=i_sender, tag=12)

    def shrinkTo(self, new_distribution_plan):
        new_matrix = ParallelMatrix(new_distribution_plan)

        for i_sender, block in enumerate(self.distributionPlan().rowDistribution()):
            i_receiver = None

            if block[0] >= new_distribution_plan.totalShape()[0]:
                break

            i_biggest_index = min(block[1], new_distribution_plan.totalShape()[0]-1)
#            print("shape new plan", new_distribution_plan.totalShape())
#            print("biggest index this block",i_biggest_index)
            for i_global_index in range(block[0], i_biggest_index+1):
                target_rank = new_distribution_plan.rankByGlobalIndex(i_global_index)
#                print("gi/tr",i_global_index, target_rank)
                if i_receiver is None:
                    i_receiver = target_rank
                    i_start_index = i_global_index
                elif i_receiver != target_rank:
#                    print("call tshrink",i_global_index, target_rank)
                    self._transferBlockShrink(i_sender, i_receiver, i_start_index, i_global_index-1, new_matrix)
                    i_start_index = i_global_index
                    i_receiver = target_rank

            if i_receiver is not None:
#                print("call broken tshrink",i_global_index, target_rank)
                self._transferBlockShrink(i_sender, i_receiver, i_start_index, i_biggest_index, new_matrix)

        return new_matrix

    def releaseMemory(self):
        self._local_matrix = None