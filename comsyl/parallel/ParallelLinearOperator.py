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

from scipy.sparse.linalg import LinearOperator
from comsyl.parallel.ParallelVector import ParallelVector

class ParallelLinearOperator(object):
    def __init__(self, parallel_matrix, parallel_vector=None):
        self._parallel_matrix = parallel_matrix

        if parallel_vector is None:
            self.setVector(ParallelVector(parallel_matrix.distributionPlan()))
        else:
            self.setVector(parallel_vector)

        # LinearOperator.__init__(self,
        #                         shape=parallel_matrix.distributionPlan().totalShape(),
        #                         matvec=self.parallelDot,
        #                         dtype=np.complex128)

    def communicator(self):
        return self._parallel_matrix.communicator()

    def setVector(self, vector):
        self._parallel_vector = vector

    def vector(self):
        return self._parallel_vector

    def __add__(self, other):
        self._parallel_matrix += other._parallel_matrix
        return self

    def __mul__(self, scalar):
        self._parallel_matrix *= scalar
        return self

    def __rmul__(self, scalar):
        self._parallel_matrix *= scalar
        return self

    def trace(self):
        return self._parallel_matrix.trace()

    def broadcastMessage(self, message=None):
        return self.communicator().bcast(message)

    def listenIfSlave(self):
        if self.communicator().Get_rank() != 0:
            while True:
                message = self.broadcastMessage()

                if message == 1:
                    self._parallelDot(None)
                else:
                    break
            return True

        return False

    def parallelDot(self, vector):
        self.broadcastMessage(1)
        self._parallelDot(vector)
        return self._parallel_vector.fullData()

    def _parallelDot(self, vector):
        self._parallel_vector.broadcast(vector, root=0)
        self._parallel_matrix.dot(self._parallel_vector)

    def finishListen(self):
        if self.communicator().Get_rank() == 0:
            self.broadcastMessage(2)

    def releaseMemory(self):
        self._parallel_matrix.releaseMemory()
        self._parallel_matrix = None
        del self._parallel_matrix
        self._parallel_vector = None
        del self._parallel_vector

class PseudoParallelLinearOperator(LinearOperator):
    def __init__(self, shape, matvec):

        self._shape = shape
        self._action = matvec

        LinearOperator.__init__(self,
                                shape=self._shape,
                                matvec=self.parallelDot,
                                dtype=np.complex128)

    def totalShape(self):
        return self._shape

    def communicator(self):
        raise NotImplementedError()

    def setVector(self, vector):
        raise NotImplementedError()

    def vector(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()

    def __mul__(self, scalar):
        raise NotImplementedError()

    def __rmul__(self, scalar):
        raise NotImplementedError()

    def trace(self):
        raise NotImplementedError()

    def broadcastMessage(self, message=None):
        raise NotImplementedError()

    def listenIfSlave(self):
        pass

    def parallelDot(self, vector):
        return self._action(vector)

    def finishListen(self):
        pass

    def releaseMemory(self):
        pass