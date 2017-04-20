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



from comsyl.autocorrelation.AutocorrelationBuilder import AutocorrelationBuilder
from comsyl.autocorrelation.AutocorrelationBuilderStrategies import BuilderStrategyPython
from comsyl.utils.Logger import log, logProgress
from comsyl.parallel.DistributionPlanPETSc import DistributionPlanPETSc
from comsyl.parallel.ParallelVector import ParallelVector
from comsyl.parallel.DistributionPlan import DistributionPlan
from comsyl.parallel.ParallelMatrixPETSc import ParallelMatrixPETSc
import mpi4py.MPI as mpi
from petsc4py import PETSc


class AutocorrelationOperator(object):
    def __init__(self, N_e, sigma_matrix, weighted_fields, x_coordinates, y_coordinates, k, number_modes):
        log("Setting up autocorrelation operator")
        self._action = 0
        self._builder = AutocorrelationBuilder(N_e, sigma_matrix, weighted_fields, x_coordinates, y_coordinates, k, strategy=BuilderStrategyPython)
        self._distribution_plan = DistributionPlan(communicator=mpi.COMM_WORLD, n_columns=self.dimensionSize(), n_rows=self.dimensionSize())
        log("Setting up PETSc interface")
        self._petSc_operator = PetScOperator(self)
        self._number_modes = number_modes
        mpi.COMM_WORLD.barrier()

    def numberModes(self):
        return self._number_modes

    def action(self, v):
        return self._builder._strategy.evaluateAllR_2_Fredholm(v)
        #return self._builder.evaluateAllR_2(v)

    def xCoordinates(self):
        return self._builder.xCoordinates()

    def yCoordinates(self):
        return self._builder.yCoordinates()

    def parrallelLinearOperator(self):
        return self

    def communicator(self):
        return self._distribution_plan.communicator()

    def parallelDot(self, v):
        v_in = ParallelVector(self._distribution_plan)
        v_in.broadcast(v, root=0)
        self.dot(v_in, v_in)
        return v_in.fullData()

    def dot(self, v_in, v_out=None):
        self._action += 1
        logProgress(self._number_modes, self._action, "Fredholm operator")

        return self._builder._strategy.evaluateAllR_2_Fredholm_parallel(v_in, v_out)

    def dimensionSize(self):
        dimension_size = self._builder._x_coordinates.shape[0] * self._builder._y_coordinates.shape[0]
        return dimension_size

    def totalShape(self):
        dimension_size = self.dimensionSize()
        shape = (dimension_size, dimension_size)
        return shape

    def distributionPlan(self):
        return self._distribution_plan

    def __rmul__(self, scalar):
        return self

    def trace(self):
        return self._builder.calculateIntensity()

    def releaseMemory(self):
        pass

    def petScMatrix(self):
        context = self._petSc_operator
        A = PETSc.Mat().createPython([self.dimensionSize(),self.dimensionSize()], context)
        A.setUp()
        return A


class PetScOperator(object):
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