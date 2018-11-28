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
from scipy.ndimage import gaussian_filter
from petsc4py import PETSc

from comsyl.mathcomsyl.utils import plot
from comsyl.mathcomsyl.Eigenmoder import Eigenmoder
from comsyl.utils.Logger import log, logProgress
from comsyl.mathcomsyl.Convolution import Convolution
from comsyl.parallel.ParallelVector import ParallelVector
from comsyl.parallel.DistributionPlanPETSc import DistributionPlanPETSc
from comsyl.parallel.DistributionPlan import DistributionPlan

class DivergenceAction(object):
    def __init__(self, x_coordinates, y_coordinates, intensity, eigenvalues_spatial, eigenvectors_parallel, phase_space_density, method):

        self._method = method

        if self._method not in ["accurate", "quick"]:
            raise Exception("Unknown divergence action %s" % self._method)

        communicator = mpi.COMM_WORLD

        n_vectors = eigenvalues_spatial.size
        self._intensity = intensity
        eigenvalues = eigenvalues_spatial
        self._number_modes = n_vectors

        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates

        self._petSc_operator = PetScOperatorDivergence(self)

        self._my_distribution_plan = DistributionPlan(communicator=communicator, n_rows=n_vectors, n_columns=self.dimensionSize())
        self._prepareEigenvectors(communicator, eigenvectors_parallel)

        self._my_eigenvalues = eigenvalues[self._my_distribution_plan.localRows()]
        self._my_eigenvectors_conjugated = self._my_eigenvectors.conj()
        
        self._my_eigenvectors_times_eigenvalues = self._my_eigenvectors
        self._my_eigenvectors = None    

        for i_e, e in enumerate(self._my_eigenvalues):
            self._my_eigenvectors_times_eigenvalues[i_e, :, :] *= e


        self._phase_space_density = phase_space_density

        self._sigma_p_x = phase_space_density.divergencePartSigmaX()
        self._sigma_p_y = phase_space_density.divergencePartSigmaY()
        self._prefactor = phase_space_density.normalizationConstant()

        log("Divergence action sigma x/y: %e %e" % (self._sigma_p_x, self._sigma_p_y))

        x_coordinates_weights = x_coordinates[(x_coordinates > -5*self._sigma_p_x) & (x_coordinates < 5*self._sigma_p_x)]
        y_coordinates_weights = y_coordinates[(y_coordinates > -5*self._sigma_p_y) & (y_coordinates < 5*self._sigma_p_y)]

        log("Calculating phase space density xy")
        weight_function = np.zeros((x_coordinates_weights.shape[0],
                                    y_coordinates_weights.shape[0]), dtype=np.complex128)

        for i_x, x in enumerate(x_coordinates_weights):
            for i_y, y in enumerate(y_coordinates_weights):
                weight_function[i_x, i_y] = phase_space_density.staticPart(np.array([x, y]))

        weight_function_horizontal = np.zeros((x_coordinates.shape[0]), dtype=np.complex128)
        weight_function_vertical = np.zeros((y_coordinates.shape[0]), dtype=np.complex128)

        log("Calculating phase space density x")
        for i_x, x in enumerate(x_coordinates_weights):
            weight_function_horizontal[i_x] = phase_space_density.staticPart(np.array([x, 0.0]))

        log("Calculating phase space density y")
        for i_y, y in enumerate(y_coordinates_weights):
            weight_function_vertical[i_y] = phase_space_density.staticPart(np.array([0.0, y]))

        #plot(x_coordinates, weight_function_horizontal)
        #plot(y_coordinates, weight_function_vertical)

        self._weight_function = weight_function
        self._weight_function_horizontal = weight_function_horizontal
        self._weight_function_vertical = weight_function_vertical
        self._i_action = 0

        self._convolution = Convolution()

    def _prepareEigenvectors(self, communicator, parallel_eigenvectors):
        distribution_plan = self._my_distribution_plan
        self._my_eigenvectors = parallel_eigenvectors.localMatrix().reshape(len(distribution_plan.localRows()),
                                                                            len(self._x_coordinates),
                                                                            len(self._y_coordinates))

    def communicator(self):
        return self._distribution_plan.communicator()

    def dimension_size(self):
        return self._weight_function.size

    def parallelDot(self, v):
        v_in = ParallelVector(self._distribution_plan)
        v_in.broadcast(v, root=0)
        self.dot(v_in, v_in)
        return v_in.fullData()

    def parrallelLinearOperator(self):
        return self

    def dot_accurate(self, v_in, v_out=None):
        if v_out is None:
            v_out = v_in

        self._i_action += 1
        eigenvalues = self._my_eigenvalues
        eigenvectors_times_eigenvalues = self._my_eigenvectors_times_eigenvalues
        c_eigenvectors = self._my_eigenvectors_conjugated

        v_r = v_in.fullData().reshape((self._x_coordinates.shape[0], self._y_coordinates.shape[0])).copy()
        res = np.zeros_like(v_r)
        tmp = np.zeros_like(v_r)

        logProgress(self._number_modes, self._i_action, "Divergence action[accurate]")
        for i in range(len(eigenvalues)):
            tmp[:, :] = c_eigenvectors[i, :, :] * v_r
            c = self._convolution.convolve2D(tmp, self._weight_function)
            tmp[:, :] = eigenvectors_times_eigenvalues[i, :, :] * c
            res[:, :] += tmp

        v_out.sumFullData(res.ravel())

    def dot_quick(self, v_in, v_out=None):

        if v_out is None:
            v_out = v_in

        self._i_action += 1

        eigenvalues = self._my_eigenvalues
        eigenvectors_times_eigenvalues = self._my_eigenvectors_times_eigenvalues
        c_eigenvectors = self._my_eigenvectors_conjugated

        v_r = v_in.fullData().reshape((self._x_coordinates.shape[0], self._y_coordinates.shape[0])).copy()
        res = np.zeros_like(v_r)
        tmp = np.zeros_like(v_r)

        sigmas = [self._sigma_p_x /(self._x_coordinates[1]-self._x_coordinates[0]),
                  self._sigma_p_y /(self._y_coordinates[1]-self._y_coordinates[0])]

        logProgress(self._number_modes, self._i_action, "Divergence action[quick]")
        for i in range(len(eigenvalues)):
            tmp[:, :] = c_eigenvectors[i, :, :] * v_r
            t_i = tmp.imag.copy()
            tmp[:, :] = gaussian_filter(tmp.real, sigmas)
            tmp[:, :] += 1j * gaussian_filter(t_i, sigmas)

            tmp[:, :] *= eigenvectors_times_eigenvalues[i, :, :]
            res[:, :] += tmp

        # no dV because solver normalizes for one integration
        normalization = 2 * np.pi*sigmas[0]*sigmas[1] * self._prefactor
        res *= normalization

        v_out.sumFullData(res.ravel())

    def dot(self, v_in, v_out=None):
        if self._method == "accurate":
            return self.dot_accurate(v_in, v_out)

        if self._method == "quick":
            return self.dot_quick(v_in, v_out)

        raise Exception("No suite able divergence method.")

    def apply(self, number_modes=None):
        eigenmoder = Eigenmoder(self._x_coordinates, self._y_coordinates)


        if number_modes is None:
            number_modes = self._number_modes-2

        twoform = eigenmoder.eigenmodes(self, number_modes)

        return twoform

    def trace(self):
        return self._intensity

    def dimensionSize(self):
        return len(self._x_coordinates) * len(self._y_coordinates)

    def totalShape(self):
        dimension_size = self.dimensionSize()
        shape = (dimension_size, dimension_size)
        return shape

    def distributionPlan(self):
        return self._distribution_plan

    def releaseMemory(self):
        pass

    def petScMatrix(self):
        context = self._petSc_operator
        A = PETSc.Mat().createPython([self.dimensionSize(),self.dimensionSize()], context)
        A.setUp()
        return A

class PetScOperatorDivergence(object):
    def __init__(self, auto_op):
        self._parent = auto_op
        self._distribution_plan = None

    def _init_vectors(self, petsc_vector):
        self._distribution_plan = DistributionPlanPETSc(communicator=mpi.COMM_WORLD, petsc_object=petsc_vector)
        self._parent._distribution_plan = self._distribution_plan
        self._vector_in = ParallelVector(self._distribution_plan)
        self._vector_out = ParallelVector(self._distribution_plan)


    def mult(self, A, x, y):
        xx = x.getArray(readonly=1)
        yy = y.getArray(readonly=0)

        if self._distribution_plan is None:
            self._init_vectors(x)

        self._vector_in.setCollective(xx)
        self._parent.dot(self._vector_in, self._vector_out)

        yy[:] = self._vector_out._local_data
