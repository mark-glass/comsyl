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

from comsyl.math.Convolution import Convolution
from comsyl.math.utils import createGaussian2D, plotSurface, norm2D

class ConvolutionTest(unittest.TestCase):

    def testConvolve2D(self):
        x = np.linspace(-6, 6, 601)
        y = np.linspace(-6, 6, 601)

        sigma_x1 = 1
        sigma_y1 = 0.5
        sigma_x2 = 1
        sigma_y2 = 0.5

        sigma_xc = np.sqrt(sigma_x1**2 + sigma_x2**2)
        sigma_yc = np.sqrt(sigma_y1**2 + sigma_y2**2)

        factor1 = createGaussian2D(sigma_x1, sigma_y1, x, y)
        factor2 = createGaussian2D(sigma_x2, sigma_y2, x, y)
        analytic_result = createGaussian2D(sigma_xc, sigma_yc, x, y)

        conv = Convolution()
        result = conv.convolve2D(factor1, factor2)

        result /= norm2D(x,y,result)
        analytic_result /= norm2D(x, y, analytic_result)

        self.assertLess(np.linalg.norm(result-analytic_result), 9e-7)

    def testAlignedConvolve2D(self):
        x = np.linspace(-6, 6, 600)
        y = np.linspace(-6, 6, 600)

        sigma_x1 = 1
        sigma_y1 = 0.5
        sigma_x2 = 1
        sigma_y2 = 0.5

        sigma_xc = np.sqrt(sigma_x1**2 + sigma_x2**2)
        sigma_yc = np.sqrt(sigma_y1**2 + sigma_y2**2)

        factor1 = createGaussian2D(sigma_x1, sigma_y1, x, y)
        factor2 = createGaussian2D(sigma_x2, sigma_y2, x, y)
        analytic_result = createGaussian2D(sigma_xc, sigma_yc, x, y)

        conv = Convolution()
        result = conv.convolve2D(factor1, factor2)
        result_aligned = conv.alignedConvolve2D(x,y,factor1, factor2)

        result /= norm2D(x,y,result)
        result_aligned /= norm2D(x,y,result_aligned)
        analytic_result /= norm2D(x, y, analytic_result)

        self.assertGreater(np.linalg.norm(result-analytic_result), 1e-1)
        self.assertLess(np.linalg.norm(result_aligned-analytic_result), 7e-7)

    def testSeperateConvolve2D(self):
        x = np.linspace(-4, 4, 101)
        y = np.linspace(-4, 4, 101)

        sigma_x1 = 1
        sigma_y1 = 0.5
        sigma_x2 = 1
        sigma_y2 = 0.5

        sigma_xc = np.sqrt(sigma_x1**2 + sigma_x2**2)
        sigma_yc = np.sqrt(sigma_y1**2 + sigma_y2**2)

        factor1 = createGaussian2D(sigma_x1, sigma_y1, x, y)
        factor2 = createGaussian2D(sigma_x2, sigma_y2, x, y)
        factor2x = np.exp(-x**2/(2*sigma_x2**2))
        factor2y = np.exp(-y**2/(2*sigma_y2**2))

        conv = Convolution()

        result = conv._seperateConvolve2D(factor1, factor2x, factor2y)
        result2 = conv.convolve2D(factor1, factor2)

        result/=norm2D(x, y, result)
        result2/=norm2D(x, y, result2)

        self.assertLess(np.linalg.norm(result-result2), 5e-14)