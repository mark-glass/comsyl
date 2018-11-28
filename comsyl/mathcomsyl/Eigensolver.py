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
from numpy.linalg import norm

class Operator(object):
    def __init__(self, f_action, dimension_size):
        self._f_action = f_action
        self._reduction_span = list()
        self._dimension_size = dimension_size

    @staticmethod
    def fromLinearOperator(linear_operator):
        return Operator(linear_operator.matvec, linear_operator.shape[0])

    def dot(self, x):
        return self._f_action(x)

    def dimensionSize(self):
        return self._dimension_size

class Eigensolver(object):
    def resizeCopy(self, array, new_first_dimension, new_second_dimension=None):

        if new_second_dimension is None:
            new_second_dimension = array.shape[1]

        resized_copy = np.zeros((new_first_dimension, new_second_dimension), dtype=array.dtype)
        resized_copy[:array.shape[0], :array.shape[1]] = array[:, :]

        return resized_copy

    def arnoldi_iteration_matprod(self, H, Q, A, number_eigenvectors):

        if Q is None or H is None:
            start_index = 1
            m = number_eigenvectors * 2 + 1
            q = self.randomVector(A.dimensionSize())
            q /= norm(q)

            Q = np.zeros((m,A.dimensionSize()), dtype=np.complex128)
            Qconj = np.zeros((m,A.dimensionSize()), dtype=np.complex128)

            H = np.zeros((m, m), dtype=np.complex128)
            Q[0, :] = q
            Qconj[0, :] = Q[0, :].conj()
        else:
            start_index = Q.shape[0]
            m = start_index + number_eigenvectors + 1

            Q = self.resizeCopy(Q, m)
            Qconj = Q.conj()
            H = self.resizeCopy(H, m, m)

        for k in range(start_index, m):
            Q[k, :] = A.dot(Q[k-1, :])
            Qconj[k, :] = Q[k,:].conj()

            H[:,k-1] = Qconj.dot(Q[k,:])

            # TODO: implement parallel leftdot.
            p=H[:, k-1]
            p[k:]=0
            th=Q.transpose().dot(p)
            #th = (H[:,k-1]).tranpose().dot(Q)
            #th = th.transpose()

            Q[k, :] -= th
            H[k, k-1] = norm(Q[k,:])

            if k%100==0:
                print("Arnoldi iteration: %i/%i"% (k, m-1))
                sys.stdout.flush()

            # Is invariant null space
            if np.abs(H[k, k-1]) < 1e-100:
                break

            Q[k,:] /= H[k, k-1]
            Qconj[k, :] = Q[k, :].conj()

        return H[0:k,0:k], Q[0:k,:]

    def arnoldi_iteration(self, H, Q, A, number_eigenvectors):

        if Q is None or H is None:
            start_index = 1
            m = number_eigenvectors * 2 + 1
            q = self.randomVector(A.dimensionSize())
            q /= norm(q)

            Q = np.zeros((m,A.dimensionSize()), dtype=np.complex128)
            Qconj = np.zeros((m,A.dimensionSize()), dtype=np.complex128)

            H = np.zeros((m, m), dtype=np.complex128)
            Q[0, :] = q
            Qconj[0, :] = Q[0, :].conj()
        else:
            start_index = Q.shape[0]
            m = start_index + number_eigenvectors + 1

            Q = self.resizeCopy(Q, m)
            Qconj = Q.conj()
            H = self.resizeCopy(H, m, m)

        for k in range(start_index, m):
            Q[k, :] = A.dot(Q[k-1, :])
            Qconj[k, :] = Q[k,:].conj()

            for j in range(k):
                H[j, k-1] = Qconj[j, :].dot(Q[k, :])
                Q[k, :] = Q[k,:] - H[j, k-1] * Q[j, :]

            H[k, k-1] = norm(Q[k,:])

            if k%100==0:
                print("Arnoldi iteration: %i/%i"% (k, m-1))
                sys.stdout.flush()

            # Is invariant null space
            if np.abs(H[k, k-1]) < 1e-100:
                break

            Q[k,:] /= H[k, k-1]
            Qconj[k, :] = Q[k, :].conj()

        return H[0:k,0:k], Q[0:k,:]

    def arnoldi(self, A, n = 25, accuracy=1e-8, accuracy_projection=None):

        n = min(A.dimensionSize(), n)

        # H: Hessenbergmatrix
        # Q: Schurbasis
        H = None
        Q = None

        for i in range(5):
            H, Q = self.arnoldi_iteration(H, Q, A, n)

            r = np.linalg.eigh(H)
            eig_val = r[0][::-1]
            eig_vec = r[1].transpose()[::-1, :]
            eig_vec = eig_vec.transpose()

            schur_vec = np.zeros((A.dimensionSize(), n), dtype=np.complex128)

            n = min(H.shape[0], n)

            for i in range(n):
                t = eig_vec[:,i]
                schur_vec[:, i] = Q.transpose().dot(t)


            residual = A.dot(schur_vec[:, n-1]/eig_val[n-1]) - schur_vec[:, n-1]
            acc = norm(residual) * np.abs(eig_val[n-1]/eig_val[0])
            acc2 = np.abs(H[-1,-2] / eig_val[n-2])

            print("Accuracy last Schur/ritz vector for normalized matrix: %e"% acc)
            print("Accuracy projection vs smallest eigenvalue: %e"% acc2)

            if accuracy_projection is not None:
                if acc2 <= accuracy_projection and acc <= accuracy:
                    print("Converged")
                    sys.stdout.flush()
                    break
            else:
                if acc <= accuracy:
                    print("Converged")
                    sys.stdout.flush()
                    break

        return eig_val[0:n], schur_vec[:,0:n].transpose()

    def randomVector(self, size):
        vector = np.random.random(size) + 0 * 1j
        return vector
