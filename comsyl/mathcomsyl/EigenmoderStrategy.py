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
import scipy.sparse.linalg as sl
from comsyl.mathcomsyl.Eigensolver import Eigensolver, Operator
from comsyl.parallel.ParallelLinearOperator import ParallelLinearOperator
from comsyl.parallel.ParallelEigensolver import ParallelEigensolver
from comsyl.parallel.ParallelVector import ParallelVector
from comsyl.parallel.ParallelMatrix import ParallelMatrix
from comsyl.parallel.DistributionPlan import DistributionPlan

from comsyl.utils.Logger import log

def crateParallelMatrixFromLocal(local_matrix):
    n_rows = local_matrix.shape[1]
    n_columns = local_matrix.shape[0]

    plan = DistributionPlan(mpi.COMM_WORLD, n_columns=n_columns, n_rows=n_rows)
    parallel_matrix = ParallelMatrix(plan)
    parallel_matrix.broadcast(local_matrix.transpose(), root=0)

    return parallel_matrix

class EigenmoderStrategy(object):
    def eigenfunctions(self, matrix, number_modes):
        raise NotImplementedError("Must override")

    def isMaster(self):
        return mpi.COMM_WORLD.Get_rank() == 0

    def isSlave(self):
        return not self.isMaster()

    def log(self, log_string):
        log(log_string)


class EigenmoderStrategyNumpy(EigenmoderStrategy):
    def eigenfunctions(self, matrix, number_modes):
        numpy_matrix = matrix.gatherMatrix(root=0)

        if self.isMaster():
            eigenvalues, eigenvectors = np.linalg.eigh(numpy_matrix)
            eigenvectors_parallel = crateParallelMatrixFromLocal(eigenvectors)
            return eigenvalues, eigenvectors_parallel
        else:
            return None, None


class EigenmoderStrategyScipy(EigenmoderStrategy):
    def eigenfunctions(self, matrix, number_modes):
        parallel_linear_operator = ParallelLinearOperator(matrix)

        if self.isSlave():
            parallel_linear_operator.listenIfSlave()
            return None, None

        self.log("Performing diagonalization using %i modes" % number_modes)

        eigenvalues, eigenvectors = sl.eigsh(parallel_linear_operator, number_modes)

        parallel_linear_operator.finishListen()

        eigenvectors_parallel = crateParallelMatrixFromLocal(eigenvectors)

        return eigenvalues, eigenvectors_parallel


class EigenmoderStrategyEigensolver(EigenmoderStrategy):
    def eigenfunctions(self, matrix, number_modes):

        if hasattr(matrix, "parrallelLinearOperator"):
            parallel_linear_operator = matrix.parrallelLinearOperator()
        else:
            parallel_linear_operator = ParallelLinearOperator(matrix)

        if self.isSlave():
            parallel_linear_operator.listenIfSlave()
            return None, None

        self.log("Performing diagonalization using %i modes" % number_modes)

        eigenvalues, eigenvectors = Eigensolver().arnoldi(Operator.fromLinearOperator(parallel_linear_operator), number_modes, 1e-6)

        parallel_linear_operator.finishListen()

        eigenvectors_parallel = crateParallelMatrixFromLocal(eigenvectors)

        return eigenvalues, eigenvectors_parallel


class EigenmoderStrategyParallelEigensolver(EigenmoderStrategy):
    def eigenfunctions(self, matrix, number_modes):
        self.log("Performing diagonalization using %i modes" % number_modes)

        eigenvalues, eigenvectors = ParallelEigensolver().arnoldi(matrix, number_modes, 1e-4)
        eigenvectors_parallel = crateParallelMatrixFromLocal(eigenvectors)

        return eigenvalues, eigenvectors_parallel


class EigenmoderStartegySLEPc(EigenmoderStrategy):
    def eigenfunctions(self, matrix, number_modes):
        import sys, slepc4py

        slepc4py.init(sys.argv)

        from petsc4py import PETSc
        from slepc4py import SLEPc

        E = SLEPc.EPS()
        E.create()

        E.setOperators(matrix.petScMatrix())
        E.setProblemType(SLEPc.EPS.ProblemType.HEP)
        #E.setType(SLEPc.EPS.Type.ARNOLDI)
        E.setFromOptions()
        E.setTolerances(tol=1e-9, max_it=200)
        E.setDimensions(nev=number_modes)
        E.solve()

        Print = PETSc.Sys.Print

        iterations = E.getIterationNumber()
        self.log("Number of iterations of the method: %d" % iterations)

        eps_type = E.getType()
        self.log("Solution method: %s" % eps_type)

        nev, ncv, mpd = E.getDimensions()
        self.log("Number of requested eigenvalues: %d" % nev)

        tol, maxit = E.getTolerances()
        self.log("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

        nconv = E.getConverged()
        self.log("Number of converged eigenpairs %d" % nconv)

        eigenvalues = np.zeros(nconv, dtype=np.complex128)
        result_vector = ParallelVector(matrix.distributionPlan())
        plan = DistributionPlan(mpi.COMM_WORLD, n_columns=matrix.totalShape()[1], n_rows=nconv)
        eigenvectors_parallel = ParallelMatrix(plan)

        # Create the results vectors
        vr, wr = matrix.petScMatrix().getVecs()
        vi, wi = matrix.petScMatrix().getVecs()
        #
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)

            result_vector.setCollective(vr.getArray())
            eigenvalues[i] = k

            if i in eigenvectors_parallel.localRows():
                eigenvectors_parallel.setRow(i, result_vector.fullData())

        return eigenvalues, eigenvectors_parallel
