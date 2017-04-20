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



import unittest

import numpy as np

from comsyl.math.Eigenmoder import SilentEigenmoder
from comsyl.math.GaussianSchellModel import GaussianSchellModel2D, GaussianSchellModel1D
from comsyl.math.Eigenmoder import Eigenmoder
from comsyl.math.MatrixBuilder import MatrixBuilder
from comsyl.math.utils import norm1D, norm2D

class TracedGSM(GaussianSchellModel2D):
    def __init__(self, A, sigma_s_x, sigma_g_x, sigma_s_y, sigma_g_y, x_coordinates, y_coordinates):

        GaussianSchellModel2D.__init__(self, A, sigma_s_x, sigma_g_x, sigma_s_y, sigma_g_y)

        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates

    def trace(self):
        data = np.zeros(self.totalShape(), dtype=np.complex128)

        for i_x, x in enumerate(self._x_coordinates):
            for i_y, y in enumerate(self._y_coordinates):
                r = np.array([x, y])
                data[i_x, i_y] = self.evaluate(r, r)

        return data

    def totalShape(self):
        return (self._x_coordinates.size, self._y_coordinates.size)

    def evaluate_r1(self, r1):
        data = np.zeros(self.totalShape(), dtype=np.complex128)

        for i_x, x in enumerate(self._x_coordinates):
            for i_y, y in enumerate(self._y_coordinates):
                r = np.array([x, y])
                data[i_x, i_y] = self.evaluate(r1, r)

        return data.flatten()


def testCoordinates(n_x=50,n_y=50):
    x = np.linspace(-3, 3, n_x)
    y = np.linspace(-1, 1, n_y)

    return x,y

fast_test = True
class EigenmoderTest(unittest.TestCase):

    @unittest.skipIf(fast_test==True, "fast")
    def testEigenmodesIntensity(self):

        f = lambda r1,r2: np.exp(-(r1[0]-r2[0])**2-0.5*(r1[1]-r2[1])**2) + 0 * 1j
        x_coordinates, y_coordinates = testCoordinates(n_x=30, n_y=30)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)
        twoform = eigenmoder.eigenmodes(f)

        for i_x_1, x_1 in enumerate(x_coordinates):
            for i_y_1, y_1 in enumerate(y_coordinates):
                r_1 = (x_1, y_1)

                r_1 = (x_1, y_1)
                r_2 = r_1

                f_r = f(r_1,r_2)
                t_r = twoform.evaluate(r_1, r_2)
                dr = f_r - t_r

                self.assertLess(abs(dr), 1e-12)

    @unittest.skipIf(fast_test==True, "fast")
    def testEigenmodesSavedIntensity(self):

        f = lambda r1,r2: np.exp(-(r1[0]-r2[0])**2-(r1[1]-r2[1])**2) + 0 * 1j
        x_coordinates, y_coordinates = testCoordinates(n_x=30, n_y=30)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)
        twoform = eigenmoder.eigenmodes(f)

        for i_x_1, x_1 in enumerate(x_coordinates):
            for i_y_1, y_1 in enumerate(y_coordinates):
                r_1 = (x_1, y_1)

                f_r = f(r_1,r_1)
                t_r = twoform.intensity()[i_x_1,i_y_1]
                dr = f_r - t_r

                self.assertLess(abs(dr), 1e-12)

    @unittest.skipIf(fast_test==True, "fast")
    def testEigenmodesAllRandom(self):

        f = lambda r1,r2: np.exp(-(r1[0]-r2[0])**2-(r1[1]-r2[1])**2) + 0 * 1j
        x_coordinates, y_coordinates = testCoordinates(n_x=30, n_y=30)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)
        twoform = eigenmoder.eigenmodes(f)

        number_samples = 8

        for i_x_1 in np.random.randint(0,len(x_coordinates),number_samples):
            x_1 = x_coordinates[i_x_1]
            for i_y_1 in np.random.randint(0,len(y_coordinates),number_samples):
                y_1 = y_coordinates[i_y_1]
                r_1 = (x_1, y_1)

                for i_x_2 in np.random.randint(0,len(x_coordinates),number_samples):
                    x_2 = x_coordinates[i_x_2]
                    for i_y_2 in np.random.randint(0,len(y_coordinates),number_samples):
                        y_2 = y_coordinates[i_y_2]
                        r_2 = (x_2, y_2)

                        f_r = f(r_1,r_2)
                        t_r = twoform.evaluate(r_1, r_2)
                        dr = f_r - t_r

                        if abs(f_r) > 10e-6:
                            self.assertLess(abs(dr), 1e-12)

    @unittest.skipIf(fast_test==True, "fast")
    def testEigenmodesAllRandomNoSymmetry(self):

        f = lambda r1,r2: np.exp(-((r1[0]-r2[0])**2)/2.0-(r1[1]-r2[1])**2) + 0 * 1j
        x_coordinates, y_coordinates = testCoordinates(n_x=30, n_y=30)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)
        twoform = eigenmoder.eigenmodes(f)

        number_samples = 8

        for i_x_1 in np.random.randint(0,len(x_coordinates),number_samples):
            x_1 = x_coordinates[i_x_1]
            for i_y_1 in np.random.randint(0,len(y_coordinates),number_samples):
                y_1 = y_coordinates[i_y_1]
                r_1 = (x_1, y_1)

                for i_x_2 in np.random.randint(0,len(x_coordinates),number_samples):
                    x_2 = x_coordinates[i_x_2]
                    for i_y_2 in np.random.randint(0,len(y_coordinates),number_samples):
                        y_2 = y_coordinates[i_y_2]
                        r_2 = (x_2, y_2)

                        f_r = f(r_1, r_2)
                        t_r = twoform.evaluate(r_1, r_2)
                        dr = f_r - t_r

                        if abs(f_r) > 10e-6:
                            self.assertLess(abs(dr), 1e-12)

    @unittest.skipIf(fast_test==True, "fast")
    def testCreateWorkMatrixAllVsByUpperHalf(self):

        f = lambda r1,r2: np.exp(-(r1[0]-r2[0])**2-(r1[1]-r2[1])**2) + (r1[0]-r2[0]) * 1j + (r1[1]-r2[1]) * 1j
        x_coordinates, y_coordinates = testCoordinates(n_x=30, n_y=30)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)

        work_matrix_all = eigenmoder._createWorkMatrixAll(f)
        work_matrix_by_upp_half = eigenmoder._createWorkMatrixByUpperHalf(f)

        diff = np.linalg.norm(work_matrix_all-work_matrix_by_upp_half)
        self.assertLess(diff,1e-10)


    @unittest.skipIf(fast_test==True, "fast")
    def testEigenmodesNormalVsFast(self):

        f = lambda r1,r2: np.exp(-(r1[0]-r2[0])**2-(r1[1]-r2[1])**2) + 0 * 1j
        x_coordinates, y_coordinates = testCoordinates(n_x=30, n_y=30)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)
        twoform_normal = eigenmoder.eigenmodes(f)
        twoform_fast = eigenmoder.eigenmodes_fast(f, None)

        number_samples = 5

        for i_x_1 in np.random.randint(0, len(x_coordinates), number_samples):
            x_1 = x_coordinates[i_x_1]
            for i_y_1 in np.random.randint(0, len(y_coordinates), number_samples):
                y_1 = y_coordinates[i_y_1]
                r_1 = (x_1, y_1)

                for i_x_2 in np.random.randint(0, len(x_coordinates), number_samples):
                    x_2 = x_coordinates[i_x_2]
                    for i_y_2 in np.random.randint(0, len(y_coordinates), number_samples):
                        y_2 = y_coordinates[i_y_2]
                        r_2 = (x_2, y_2)

                        f_r = twoform_fast.evaluate(r_1, r_2)
                        t_r = twoform_normal.evaluate(r_1, r_2)
                        dr = f_r - t_r

                        if abs(f_r) > 10e-8:
                            self.assertLess(abs(dr), 1e-8)

    @unittest.skipIf(fast_test==True, "fast")
    def testEigenmoder1DX(self):
        x_coordinates = np.linspace(-3, 3, 100)
        y_coordinates = np.array([0.0])

        f = lambda r1,r2: np.exp(-(r1[0]-r2[0])**2)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)
        twoform = eigenmoder.eigenmodes(f)

        y = [twoform.evaluate([x, 0.0], [1, 0.0]) for x in x_coordinates]
        y_f = [f([x, 0.0], [1, 0.0]) for x in x_coordinates]
        d_y = np.array(y)-np.array(y_f)

        self.assertLess(np.linalg.norm(d_y), 1e-8)

    @unittest.skipIf(fast_test==True, "fast")
    def testEigenmoder1DY(self):
        x_coordinates = np.array([0.0])
        y_coordinates = np.linspace(-3, 3, 100)

        f = lambda r1,r2: np.exp(-(r1[0]-r2[0])**2)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)
        twoform = eigenmoder.eigenmodes(f)

        x = [twoform.evaluate([0.0, y], [0.0, 1.0]) for y in y_coordinates]
        x_f = [f([0.0, y], [0.0, 1.0]) for y in y_coordinates]
        d_x = np.array(x)-np.array(x_f)

        self.assertLess(np.linalg.norm(d_x), 1e-8)

    @unittest.skipIf(fast_test==True, "fast")
    def testGaussianSchellModel1D(self):
        x_coordinates = np.linspace(-10, 10, 100)
        y_coordinates = np.array([0.0])

        gsm = GaussianSchellModel1D(1.0,1.0,1.0)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)

        f = lambda r1,r2: gsm.evaluate(r1[0], r2[0])

        twoform = eigenmoder.eigenmodes(f)

        for n in range(min(len(twoform.eigenvalues()), 15)):
            y = twoform.eigenvectors()[n,:,0]
            phi = [gsm.phi(n, x) for x in x_coordinates]

            phi /= np.linalg.norm(phi)

            diff_norm_eig = np.sum(np.abs(phi-y))
            diff_norm_neg = np.sum(np.abs(-1*phi-y))

            if diff_norm_eig > diff_norm_neg:
                phi *= -1

            diff_norm = np.sum(np.abs(phi-y))

            self.assertLess(diff_norm, 5 * 1e-10)

    @unittest.skipIf(fast_test==True, "fast")
    def testGaussianSchellModel1DX(self):
        x_coordinates = np.linspace(-12, 12, 1000)
        y_coordinates = np.array([0.0])

        gsm = GaussianSchellModel2D(A=1,
                                    sigma_s_x=1.5,
                                    sigma_g_x=1.5,
                                    sigma_s_y=10000000,
                                    sigma_g_y=10000000)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)

        f = lambda r1,r2: gsm.evaluate(r1, r2)

        twoform = eigenmoder.eigenmodes(f)

        for n in range(min(len(twoform.eigenvalues()), 15)):
            y = twoform.eigenvectors()[n, :, 0]
            phi = gsm.phi_nm(n, 0, x_coordinates, [0.0])[:, 0]

            phi /= np.linalg.norm(phi)

            diff_norm_eig = np.sum(np.abs(phi-y))
            diff_norm_neg = np.sum(np.abs(-1*phi-y))

            if diff_norm_eig > diff_norm_neg:
                phi *= -1

            diff_norm = np.sum(np.abs(phi-y))

            self.assertLess(diff_norm, 1e-8)

    @unittest.skipIf(fast_test==True, "fast")
    def testGaussianSchellModel1DY(self):
        x_coordinates = np.array([0.0])
        y_coordinates = np.linspace(-15, 15, 100)


        gsm = GaussianSchellModel2D(A=1,
                                    sigma_s_x=10000000,
                                    sigma_g_x=10000000,
                                    sigma_s_y=2.0,
                                    sigma_g_y=1.5)

        eigenmoder = SilentEigenmoder(x_coordinates, y_coordinates)

        f = lambda r1,r2: gsm.evaluate(r1, r2)

        twoform = eigenmoder.eigenmodes(f)

        for n in range(min(len(twoform.eigenvalues()), 15)):
            x = twoform.eigenvectors()[n, 0, :]
            x = x / np.sqrt(np.trapz(np.abs(x)**2, y_coordinates))
            phi = np.array([gsm._mode_y.phi(n, y) for y in y_coordinates])

            diff_norm_eig = np.sum(np.abs(phi-x))
            diff_norm_neg = np.sum(np.abs(-1*phi-x))

            if diff_norm_eig > diff_norm_neg:
                phi *= -1

            diff_norm = np.sum(np.abs(phi-x))

            self.assertLess(diff_norm, 1 * 1e-8)

    @unittest.skipIf(fast_test==True, "fast")
    def testGaussianSchellModel2DSliceX(self):
        x_coordinates = np.linspace(-12, 12, 200)
        y_coordinates_all = np.linspace(-10, 10, 50)

        gsm = GaussianSchellModel2D(A=1,
                                    sigma_s_x=1.5,
                                    sigma_g_x=1.3,
                                    sigma_s_y=1.0,
                                    sigma_g_y=1.0)

        for y_coordinates in y_coordinates_all:

            eigenmoder = Eigenmoder(x_coordinates, [y_coordinates])

            builder = MatrixBuilder(x_coordinates, [y_coordinates])
            builder._mode_element_wise = True
            #work_matrix = builder._createParallelMatrix(gsm.evaluate_r1)
            work_matrix = builder._createParallelMatrix(gsm.evaluate)
            twoform = eigenmoder.eigenmodes(work_matrix)

            for n in range(min(len(twoform.eigenvalues()), 15)):
                y = twoform.eigenvectors()[n, :, 0]
                phi = gsm.phi_nm(n, 0, x_coordinates, [0.0])[:, 0]

                phi /= np.trapz(phi**2,x_coordinates) **0.5
                y /= np.trapz(y**2,x_coordinates)**0.5

                diff_norm_eig = np.sum(np.abs(phi-y))
                diff_norm_neg = np.sum(np.abs(-1*phi-y))

                if diff_norm_eig > diff_norm_neg:
                    phi *= -1

                import matplotlib.pyplot as plt
                #plt.plot(x_coordinates, phi)
                #plt.plot(x_coordinates, y)
                #plt.show()


                diff_norm = np.sum(np.abs(phi-y))
                print(diff_norm)
                self.assertLess(diff_norm, 5 * 1e-10)

    @unittest.skipIf(fast_test==True, "fast")
    def testGaussianSchellModel2DYSliceY(self):
        x_coordinates_all = np.linspace(-10, 10, 50)
        y_coordinates = np.linspace(-12, 12, 50)


        for x_coordinates in x_coordinates_all:

            gsm = GaussianSchellModel2D(A=1,
                                        sigma_s_x=1.0,
                                        sigma_g_x=1.0,
                                        sigma_s_y=1.2,
                                        sigma_g_y=1.6)

            eigenmoder = Eigenmoder([x_coordinates], y_coordinates)

            builder = MatrixBuilder([x_coordinates], y_coordinates)
            builder._mode_element_wise = True
            #work_matrix = builder._createParallelMatrix(gsm.evaluate_r1)
            work_matrix = builder._createParallelMatrix(gsm.evaluate)
            twoform = eigenmoder.eigenmodes(work_matrix)

            for n in range(min(len(twoform.eigenvalues()), 10)):
                x = twoform.eigenvectors()[n, 0, :]
                x = x / norm1D(y_coordinates, x)
                phi = np.array([gsm._mode_y.phi(n, y) for y in y_coordinates])

                phi /= np.trapz(phi**2, y_coordinates) **0.5
                x /= np.trapz(x**2, y_coordinates)**0.5

                diff_norm_eig = np.sum(np.abs(phi-x))
                diff_norm_neg = np.sum(np.abs(-1*phi-x))

                if diff_norm_eig > diff_norm_neg:
                    phi *= -1

                diff_norm = norm1D(y_coordinates, phi-x)

                print(diff_norm)
                self.assertLess(diff_norm, 1 * 1e-8)

    def TwoDIntegral(self, x_coords, y_coords, i):
        s = np.zeros(i.shape[0])

        for k in range(i.shape[0]):
            s[k]=np.trapz(i[k, :], y_coords)

        res = np.trapz(s, x_coords)

        return res

    #@unittest.skipIf(fast_test==True, "fast")
    def testGaussianSchellModel(self):
        n_points = 61
        x_coordinates = np.linspace(-10, 10, n_points)
        y_coordinates = np.linspace(-10, 10, n_points)

        eigenmoder = Eigenmoder(x_coordinates, y_coordinates)

        gsm = TracedGSM(A=1,
                        sigma_s_x=0.9,
                        sigma_g_x=1.0,
                        sigma_s_y=1.5,
                        sigma_g_y=1.5,
                        x_coordinates=x_coordinates,
                        y_coordinates=y_coordinates)

        eigenvalues_x = np.array([gsm._mode_x.beta(i) for i in range(n_points)])
        eigenvalues_y = np.array([gsm._mode_y.beta(i) for i in range(n_points)])

        i_c = 0
        array_size = eigenvalues_x.shape[0]*eigenvalues_y.shape[0]
        f=np.zeros(array_size, dtype=np.complex128)
        ind=np.zeros((array_size,2), dtype=np.int)
        for i_x, e_x in enumerate(eigenvalues_x):
            for i_y, e_y in enumerate(eigenvalues_y):
                f[i_c] = e_x * e_y
                ind[i_c, :] = (i_x, i_y)
                i_c += 1

        ind_sort = f.argsort()[::-1]


        builder = MatrixBuilder(x_coordinates, y_coordinates)
        builder._mode_element_wise = True
        #work_matrix = builder._createParallelMatrix(gsm.evaluate_r1)
        work_matrix = builder._createParallelMatrix(gsm.evaluate)


        twoform = eigenmoder.eigenmodes(work_matrix)
        #twoform.save("gsm2dtest.npz")
        x_index = np.abs(x_coordinates).argmin()
        y_index = np.abs(y_coordinates).argmin()
        for i_e, eigenmode in enumerate(twoform.eigenvectors()):

            # t = np.zeros_like(eigenmode)
            # t[:-1, :-1] = eigenmode[1:, 1:]
            # eigenmode = t



            n, m = ind[ind_sort[i_e],:]
            mode = gsm.phi_nm(n, m, x_coordinates, y_coordinates)

            norm_mode = norm2D(x_coordinates, y_coordinates, mode)
            norm_eigenmode = norm2D(x_coordinates, y_coordinates, eigenmode)

            print(norm_mode, norm_eigenmode)

            norm_mode = mode.max()
            mode /= norm_mode

            normed_eigenmode = eigenmode / eigenmode.max()#norm_eigenmode

            diff_norm_eig = np.sum(np.abs(mode-normed_eigenmode))
            diff_norm_neg = np.sum(np.abs(-1*mode-normed_eigenmode))

            if diff_norm_eig > diff_norm_neg:
                normed_eigenmode *= -1

            normed_eigenmode /= np.abs(normed_eigenmode).max()
            mode /= np.abs(mode).max()


            # import matplotlib.pyplot as plt
            # plt.plot(x_coordinates, mode[:, y_index])
            # plt.plot(x_coordinates, normed_eigenmode[:, y_index])
            # plt.show()
            # plt.plot(y_coordinates, mode[x_index, :])
            # plt.plot(y_coordinates, normed_eigenmode[x_index, :])
            # plt.show()


            # from comsyl.math.utils import plotSurface
            # plotSurface(x_coordinates, y_coordinates, mode)
            # plotSurface(x_coordinates, y_coordinates, normed_eigenmode)
            # plotSurface(x_coordinates, y_coordinates, mode - normed_eigenmode)

            diff_norm = norm2D(x_coordinates, y_coordinates,normed_eigenmode-mode)
            print("%i: diff norm (n:%i, m: %i); beta %e %e" % (i_e,n,m,gsm.beta(n,m),twoform.eigenvalues()[i_e]), diff_norm)


            if i_e<50:
                #self.assertLess(diff_norm, 1e-8)
                if diff_norm>1e-8:

                  from comsyl.math.utils import plotSurface
                  plotSurface(x_coordinates, y_coordinates, mode)
                  plotSurface(x_coordinates, y_coordinates, normed_eigenmode)
                  plotSurface(x_coordinates, y_coordinates, mode - normed_eigenmode)

                  import matplotlib.pyplot as plt
                  plt.plot(x_coordinates, mode[:, y_index])
                  plt.plot(x_coordinates, normed_eigenmode[:, y_index])
                  plt.show()
                  plt.plot(y_coordinates, mode[x_index, :])
                  plt.plot(y_coordinates, normed_eigenmode[x_index, :])
                  plt.show()

    def testDecompositionSinc(self):
        x_coordinates = np.linspace(-15, 15, 400)
        y_coordinates_all = np.linspace(-10, 10, 1)

        for y_coordinates in y_coordinates_all:

            eigenmoder = Eigenmoder(x_coordinates, [y_coordinates])

            builder = MatrixBuilder(x_coordinates, [y_coordinates])
            builder._mode_element_wise = True
            #work_matrix = builder._createParallelMatrix(gsm.evaluate_r1)
            f = lambda x,y: np.sinc(x[0]**2+y[0]**2)
            work_matrix = builder._createParallelMatrix(f)
            twoform = eigenmoder.eigenmodes(work_matrix)

            print(twoform.evaluate(np.array([0,0]),np.array([0,0])), np.sinc(0))

            print(twoform.evaluate(np.array([0.1**0.5,0]),np.array([0,0])), np.sinc(0.1))

            for n in range(min(len(twoform.eigenvalues()), 15)):
                y = twoform.eigenvectors()[n, :, 0]
                eig = twoform.eigenvalues()[n]

                print(eig)

                import matplotlib.pyplot as plt
                plt.plot(x_coordinates, y)
                plt.show()

