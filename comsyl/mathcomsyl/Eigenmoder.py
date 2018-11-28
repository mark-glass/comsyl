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
import scipy
import mpi4py.MPI as mpi
from comsyl.mathcomsyl.Twoform import Twoform
from comsyl.mathcomsyl.TwoformVectors import TwoformVectorsParallelMatrix

from comsyl.parallel.ParallelLinearOperator import ParallelLinearOperator

from comsyl.mathcomsyl.EigenmoderStrategy import EigenmoderStartegySLEPc

from comsyl.utils.Logger import log

class Eigenmoder(object):
    def __init__(self, coordinates_x, coordinates_y):
        self._coordinates_x = coordinates_x
        self._coordinates_y = coordinates_y

        if len(coordinates_x)>1 and len(coordinates_y)>1:
            self._grid_area = (coordinates_x[1] - coordinates_x[0]) * (coordinates_y[1] - coordinates_y[0])
        else:
            self._grid_area = 1

        self._strategy = EigenmoderStartegySLEPc()

    def isMaster(self):
        return mpi.COMM_WORLD.Get_rank() == 0

    def isSlave(self):
        return not self.isMaster()

    def _determineNumberModes(self, max_number_modes, number_modes):
        if(number_modes is None):
            number_modes = max_number_modes-2
        else:
            number_modes = min(max_number_modes-2,number_modes)

        return number_modes

    def eigenmodes(self, work_matrix, number_modes=25, do_not_gather=False):
        diagonal_elements = work_matrix.trace()

        number_modes = self._determineNumberModes(work_matrix.totalShape()[0], number_modes)

        self.log("Performing diagonalization for %i modes" % number_modes)
        eigenvalues, eigenvectors_parallel = self._strategy.eigenfunctions(work_matrix, number_modes)
        self.log("done")

        self.log("Determine eigenvector accuracy")
        eigenvector_errors = self.determineEigenfunctionErrors(work_matrix, eigenvalues, eigenvectors_parallel)
        self.log("done")

        self.log("Release matrix memory")
        work_matrix.releaseMemory()
        self.log("done")

        # Correct for equivalent norm
        eigenvectors_parallel *= (1/self._grid_area**0.5)
        eigenvalues *= self._grid_area

        if do_not_gather:
            self.log("Returning distributed eigenbasis.")
            return eigenvalues, eigenvectors_parallel

        if hasattr(work_matrix, "xCoordinates"):
            x = work_matrix.xCoordinates()
            y = work_matrix.yCoordinates()
        else:
            x = self._coordinates_x
            y = self._coordinates_y

        twoform_vectors = TwoformVectorsParallelMatrix(x, y, parallel_matrix=eigenvectors_parallel)
        twoform = Twoform(x, y, diagonal_elements, eigenvalues, twoform_vectors)
        twoform.setEigenvectorErrors(eigenvector_errors)

        return twoform

    def determineEigenfunctionErrors(self, matrix, eigenvalues, eigenfunctions):

        if hasattr(matrix, "parrallelLinearOperator"):
            parallel_linear_operator = matrix.parrallelLinearOperator()
        else:
            parallel_linear_operator = ParallelLinearOperator(matrix)

        indices_to_determine = []
        for i in range(len(eigenvalues)):
            if int(len(eigenvalues)*0.1) != 0:
                if i % int(len(eigenvalues)*0.1) == 0:
                    indices_to_determine.append(i)
            else:
                indices_to_determine.append(i)

        errors = np.zeros((len(indices_to_determine),3))

        self.log("Mode   abs error     normalized error")
        for c_i, i in enumerate(indices_to_determine):
            eigenfunction = eigenfunctions.globalRow(i)
            v_out = parallel_linear_operator.parallelDot(eigenfunction)
            l2_error = scipy.linalg.norm(v_out/eigenvalues[i] - eigenfunction)
            max_error = np.abs(v_out - eigenvalues[i] * eigenfunction).max()/eigenvalues[i].real
            normalized_l2_error = max_error

            errors[c_i, :] = i, l2_error, normalized_l2_error

            self.log("%i.    %e %e" %(i, l2_error, normalized_l2_error))

        return errors

    def log(self, log_string):
        log(log_string)

class SilentEigenmoder(Eigenmoder):
    def __init__(self, coordinates_x, coordinates_y, mode_single=True):
        Eigenmoder.__init__(self, coordinates_x, coordinates_y, mode_single)

    def _printProgress(self, n_coordinates, index):
        return