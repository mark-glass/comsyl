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

class DistributionPlan(object):
    def __init__(self, communicator, n_columns, n_rows, strategy="rowblocks"):
        self._communicator = communicator
        self._n_rows = n_rows
        self._n_columns = n_columns
        self._strategy = strategy

    def communicator(self):
        return self._communicator

    def numberRanks(self):
        return self._communicator.Get_size()

    def ranks(self):
        return np.arange(self.numberRanks())

    def myRank(self):
        n_rank = self._communicator.Get_rank()
        return n_rank

    def _rowsForRank(self):
        rows = np.arange(self._n_rows)
        rows_for_rank = np.array_split(rows, self.numberRanks())

        return rows_for_rank

    def rowDistribution(self):
        rows_for_rank = self._rowsForRank()

        row_distribution = list()
        for rows in rows_for_rank:
            row_distribution.append((rows.min(), rows.max()))

        return row_distribution

    def rows(self, rank):
        rows_for_rank = self._rowsForRank()
        return rows_for_rank[rank]

    def rankByGlobalIndex(self, global_index):
        for i_rank in self.ranks():
            if global_index in self.rows(i_rank):
                return i_rank

        raise Exception("Global index not allocated to any rank.")

    def columns(self, rank):
        return np.arange(self._n_columns)

    def shape(self, rank):
        return (len(self.rows(rank)), len(self.columns(rank)))


    def localRows(self):
        return self.rows(self.myRank())

    def localColumns(self):
        return self.columns(self.myRank())

    def localShape(self):
        return self.shape(self.myRank())

    def totalShape(self):
        return (self._n_rows, self._n_columns)

    def globalToLocalIndex(self, global_index):
        local_rows = self.localRows()

        local_index = global_index - local_rows[0]

        if local_index < 0:
            raise Exception("Negative index")

        if local_index >= len(local_rows):
            raise Exception("%i: Index too large (global/local/number local rows): %i %i %i" % (self.myRank(), global_index, local_index, len(local_rows)))

        return local_index

    def localToGlobalIndex(self, local_index):
        local_rows = self.localRows()

        if local_index < 0:
            raise Exception("Negative index")

        if local_index >= len(local_rows):
            raise Exception("Index too large %i %i" % (local_index, len(local_rows)) )

        global_index = local_rows[0] + local_index

        return global_index

    def __eq__(self, other):
#        return self is other
        if not self.communicator() is other.communicator():
            return False

        if not self._n_rows == other._n_rows:
            return False

        if not self._n_columns == other._n_columns:
            return False

        if not self._strategy == other._strategy:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)