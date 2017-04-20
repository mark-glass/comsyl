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

from comsyl.autocorrelation.SigmaMatrix import SigmaWaist


def createStraightSectionTestMatrix():
    sigma_matrix = SigmaWaist(sigma_x=3e-6,
                                    sigma_y=1e-6,
                                    sigma_x_prime=5e-6,
                                    sigma_y_prime=4e-6)
    return sigma_matrix


class SigmaMatrixText(unittest.TestCase):
    def testSigmaPpDeterminant(self):

        sigma_matrix = createStraightSectionTestMatrix()

        determinant = 1.0/((sigma_matrix.sigma_x_prime() * sigma_matrix.sigma_y_prime()) ** -2)

        diff = np.abs(determinant - sigma_matrix.SigmaPpDeterminant())

        self.assertLess(diff, 1e-12)

    def testSigmaInverseDeterminant(self):

        sigma_matrix = createStraightSectionTestMatrix()

        determinant = (sigma_matrix.sigma_x() * sigma_matrix.sigma_y() *
                       sigma_matrix.sigma_x_prime() * sigma_matrix.sigma_y_prime()) ** -2

        # In straight section determinant the delta-z sub determinante is set to equal cancel 2pi

        ratio = np.abs((2*np.pi *determinant) / sigma_matrix.SigmaInverseDeterminant())

        self.assertLess(np.abs(1-ratio), 1e-12)