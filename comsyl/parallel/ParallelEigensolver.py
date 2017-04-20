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



import sys
import numpy as np
import scipy.linalg
from numpy.linalg import norm

from comsyl.parallel.DistributionPlan import DistributionPlan
from comsyl.parallel.ParallelMatrix import ParallelMatrix
from comsyl.parallel.ParallelVector import ParallelVector

import mpi4py.MPI as mpi

class ParallelEigensolver(object):
    def resizeCopy(self, array, new_first_dimension, new_second_dimension=None):

        if new_second_dimension is None:
            new_second_dimension = array.shape[1]

        resized_copy = np.zeros((new_first_dimension, new_second_dimension), dtype=array.dtype)
        resized_copy[:array.shape[0], :array.shape[1]] = array[:, :]

        return resized_copy

    def _createIterationMatrices(self, H, Q, A, number_eigenvectors):
        if Q is None or H is None:
            start_index = 1
            m = number_eigenvectors * 2 + 1
            q = self.randomVector(A.totalShape()[0])
            q /= norm(q)

            distribution_plan = DistributionPlan(mpi.COMM_WORLD, n_rows=m, n_columns=A.totalShape()[0])

            Q = ParallelMatrix(distribution_plan)

            H = np.zeros((m, m), dtype=np.complex128)
            if 0 in Q.localRows():
                Q.setRow(0, q)
        else:
            start_index = Q.totalShape()[0]
            m = start_index + number_eigenvectors + 1

            new_distribution_plan = DistributionPlan(mpi.COMM_WORLD, n_rows=m, n_columns=A.totalShape()[0])

            # if new_distribution_plan.totalShape()[0] <= Q.distributionPlan().totalShape()[0]:
            #     new_distribution_plan = Q.distributionPlan()

            Q = Q.enlargeTo(new_distribution_plan)
            H = self.resizeCopy(H, m, m)

        return H, Q, start_index, m

    def arnoldi_iteration(self, H, Q, A, number_eigenvectors):
        H, Q, start_index, m = self._createIterationMatrices(H, Q, A, number_eigenvectors)

        qt_distribution_plan = DistributionPlan(mpi.COMM_WORLD, n_rows=A.totalShape()[0], n_columns=m)
        q_distribution_plan = DistributionPlan(mpi.COMM_WORLD, n_rows=m, n_columns=A.totalShape()[0])

        parallel_vector = ParallelVector(qt_distribution_plan)

        parallel_vector._full_data[:] = Q.globalRow(start_index-1)

        for k in range(start_index, m):
            A.dot(parallel_vector, parallel_vector)

            if k in Q.localRows():
                Q.setRow(k, parallel_vector.fullData())


            q_k = Q.globalRow(k)
            if k == m or True:
                for j in range(k):
                    q_j = Q.cachedGlobalRow(j)
                    H[j, k-1] = np.vdot(q_j, q_k)
                    q_k -= H[j, k-1] * q_j
            # else:
            #     pv = ParallelVector(qt_distribution_plan)
            #     pv2 = ParallelVector(q_distribution_plan)
            #     pv._full_data[:] = q_k
            #     Q.dot(pv,pv2,complex_conjugate=True)
            #     H[:, k-1] = pv2.fullData()[:]
            #
            #     p=H[:, k-1]
            #     p[k:]=0
            #     pv2._full_data[:] = p
            #     Q.dotForTransposed(pv2, pv)
            #
            #     q_k -= pv.fullData()
            #     H[k, k-1] = norm(q_k)

            Q.resetCacheGlobalRow()

            if k in Q.localRows():
                Q.setRow(k, q_k)

            row_data = Q.globalRow(k)
            H[k, k-1] = norm(row_data)

            norm_row_data = row_data / H[k, k-1]


            if k%100==0 and Q.distributionPlan().myRank()==0:
                print("Arnoldi iteration: %i/%i"% (k, m-1))
                sys.stdout.flush()

            # Is invariant null space
            if np.abs(H[k, k-1]) < 1e-100:
                break

            if k in Q.localRows():
                Q.setRow(k, norm_row_data)


            parallel_vector._full_data[:]= norm_row_data

        new_distribution_plan = DistributionPlan(mpi.COMM_WORLD, n_rows=k, n_columns=A.totalShape()[1])
        Q = Q.shrinkTo(new_distribution_plan)

        return H[0:k,0:k], Q

    def arnoldi(self, A, n = 25, accuracy=1e-8, accuracy_projection=None):

        n = min(A.totalShape()[0], n)

        # H: Hessenbergmatrix
        # Q: Schurbasis
        H = None
        Q = None

        my_rank = A.distributionPlan().myRank()

        for i in range(5):
            H, Q = self.arnoldi_iteration(H, Q, A, n)

            r = np.linalg.eigh(H)
            eig_val = r[0][::-1]
            eig_vec = r[1].transpose()[::-1, :]
            eig_vec = eig_vec.transpose()

            schur_vec = np.zeros((A.totalShape()[0], n), dtype=np.complex128)

            n = min(H.shape[0], n)

            q_distribution_plan = Q.distributionPlan()
            qt_distribution_plan = DistributionPlan(mpi.COMM_WORLD, n_rows=Q.totalShape()[1], n_columns=Q.totalShape()[0])

            parallel_vector_in = ParallelVector(q_distribution_plan)
            parallel_vector_out = ParallelVector(qt_distribution_plan)
            for i in range(n):
                t = eig_vec[:,i]
                full_data = np.append(eig_vec[:,i], np.zeros(Q.totalShape()[0]-eig_vec[:,i].shape[0]))
                parallel_vector_in._full_data[:] = full_data
                Q.dotForTransposed(parallel_vector_in, parallel_vector_out)
                schur_vec[:, i] = parallel_vector_out.fullData()


            parallel_vector_out._full_data[:] = schur_vec[:, n-1]

            A.dot(parallel_vector_out)
            acc = scipy.linalg.norm( parallel_vector_out.fullData()/eig_val.max() - (eig_val[n-1]/eig_val.max()) * schur_vec[:, n-1])

            acc2 = np.abs(H[-1, -2] / eig_val[n-2])

            if my_rank == 0:
                print("Accuracy last Schur/ritz vector for normalized matrix: %e"% acc)
                print("Accuracy projection vs smallest eigenvalue: %e"% acc2)

            if accuracy_projection is not None:
                if acc2 <= accuracy_projection and acc <= accuracy:
                    if my_rank == 0:
                        print("Converged")
                        sys.stdout.flush()
                    break
            else:
                if acc <= accuracy:
                    if my_rank == 0:
                        print("Converged")
                    sys.stdout.flush()
                    break

        return eig_val[0:n], schur_vec[:,0:n]

    def randomVector(self, size):
        vector = np.random.random(size) + 0 * 1j
        return vector
