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

from comsyl.math.GaussianSchellModel import GaussianSchellModel2D
from comsyl.math.Eigenmoder import Eigenmoder
from comsyl.math.utils import norm2D

from comsyl.data.AutocorrelationFunctionLoader import AutocorrelationFunctionLoader


def getTwoformGSM(n_points=50):
        x_coordinates = np.linspace(-10, 10, n_points)
        y_coordinates = np.linspace(-10, 10, n_points)

        eigenmoder = Eigenmoder(x_coordinates, y_coordinates)

        gsm = GaussianSchellModel2D(A=1,
                                    sigma_s_x=1.0,
                                    sigma_g_x=1.0,
                                    sigma_s_y=1.5,
                                    sigma_g_y=1.5)

        twoform = eigenmoder.eigenmodes(gsm.evaluate)
        return twoform

class TwoformTest(unittest.TestCase):

    @unittest.expectedFailure
    def testCachedYEvaluate(self):
        a = AutocorrelationFunctionLoader().create("new_u18_2m_1h_s1.0.npz")
        twoform = a.Twoform()

        x_coordinates = twoform.xCoordinates()[::5]
        y_coordinates = twoform.yCoordinates()[::5]

        offset = np.array([0.0, y_coordinates[3]])

        for i_y, y in enumerate(y_coordinates):
            for i_x, x in enumerate(x_coordinates):
                r_1 = np.array([offset[0]+x, offset[1]+y])
                r_2 = np.array([offset[0]-x, offset[1]-y])

                cached_value = twoform.cachedYEvaluate(r_1, r_2)
                direct_value = twoform.evaluate(r_1, r_2)
                diff = np.abs(cached_value-direct_value)
                self.assertLess(diff, 1e-4)

    def testDot(self):
        a = AutocorrelationFunctionLoader().create("new_u18_2m_1h_s1.0.npz")
        twoform = a.Twoform()

        v = twoform.vector(0)
        v2 = twoform.dot(v)
        v_norm = norm2D(twoform.xCoordinates(), twoform.yCoordinates(), v)
        v2_norm = norm2D(twoform.xCoordinates(), twoform.yCoordinates(), v2)

        self.assertLess(v_norm - v2_norm/twoform.eigenvalues()[0], 1e-6)

        diff_norm = norm2D(twoform.xCoordinates(), twoform.yCoordinates(), v2/twoform.eigenvalues()[0] - v)
        self.assertLess(diff_norm, 1e-7)

    def testEvaluateInversionCut(self):
        a = AutocorrelationFunctionLoader().create("new_u18_2m_1h_s1.0.npz")
        twoform = a.Twoform()

        inversion_cut = twoform.evaluateInversionCut()

        for x in twoform.xCoordinates():
            r = np.array([x, 0])
            res_e = twoform.evaluate(r, -r)
            i_x = twoform.xIndexByCoordinate(x)
            i_y = twoform.yIndexByCoordinate(0.0)

            self.assertLess(np.abs(res_e-inversion_cut[i_x, i_y]), 1e-14)

        for y in twoform.yCoordinates():
            r = np.array([0, y])
            res_e = twoform.evaluate(r, -r)
            i_x = twoform.xIndexByCoordinate(0.0)
            i_y = twoform.yIndexByCoordinate(y)
            self.assertLess(np.abs(res_e-inversion_cut[i_x, i_y]), 1e-14)

    def testEvaluateVCut(self):
        a = AutocorrelationFunctionLoader().create("new_u18_2m_1h_s1.0.npz")
        twoform = a.Twoform()

        inversion_cut = twoform.evaluateVCut()

        for x in twoform.xCoordinates():
            r = np.array([x, 0])
            res_e = twoform.evaluate(r, -r)
            i_x = twoform.xIndexByCoordinate(x)
            i_y = twoform.yIndexByCoordinate(0.0)

            self.assertLess(np.abs(res_e-inversion_cut[i_x, i_y]), 1e-14)

        for y in twoform.yCoordinates():
            r = np.array([0, y])
            res_e = twoform.evaluate(r, r)
            i_x = twoform.xIndexByCoordinate(0.0)
            i_y = twoform.yIndexByCoordinate(y)
            self.assertLess(np.abs(res_e-inversion_cut[i_x, i_y]), 1e-14)

    def testEvaluateSmallerCut(self):
        a = AutocorrelationFunctionLoader().create("new_u18_2m_1h_s1.0.npz")
        twoform = a.Twoform()

        inversion_cut = twoform.evaluateSmallerCut()

        for x in twoform.xCoordinates():
            r = np.array([x, 0])
            res_e = twoform.evaluate(r, r)
            i_x = twoform.xIndexByCoordinate(x)
            i_y = twoform.yIndexByCoordinate(0.0)

            self.assertLess(np.abs(res_e-inversion_cut[i_x, i_y]), 1e-14)

        for y in twoform.yCoordinates():
            r = np.array([0, y])
            res_e = twoform.evaluate(r, -r)
            i_x = twoform.xIndexByCoordinate(0.0)
            i_y = twoform.yIndexByCoordinate(y)
            self.assertLess(np.abs(res_e-inversion_cut[i_x, i_y]), 1e-14)