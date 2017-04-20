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

from srwlib import *

from comsyl.autocorrelation.AutocorrelationFunction import AutocorrelationFunction
from comsyl.autocorrelation.AutocorrelationFunctionPropagator import AutocorrelationFunctionPropagator


def createBeamline():
    optBL = SRWLOptC([SRWLOptD(20),SRWLOptA('r', 'a', 0.0008, 0.0005),SRWLOptD(20),SRWLOptA('r', 'a', 0.000015, 0.000015)],
                     [[1,  1, 3.0,  0,  0, 1.0, 1.0, 1.0, 1.0,  0,  0,   0],
                      [0,  1, 4.0,  0,  0, 1.0, 1.0, 1.0, 1.0,  0,  0,   0],
                      [1,  1, 2.0,  0,  0, 1.0, 1.0, 1.0, 1.0,  0,  0,   0],
                      [0,  1, 1.0,  0,  0, 1.0, 1.0, 1.0, 1.0,  0,  0,   0]])

    return optBL

def createBeamline2():
    optBL = SRWLOptC([SRWLOptD(1.0)],
                     [[1,  1, 4.0,  0,  0, 1.0, 1.0, 1.0, 1.0,  0,  0,   0],
                      [0,  0, 1.0,  0,  0, 0.5, 1.0, 0.5, 1.0,  0,  0,   0]])

    return optBL


class AutocorrelationFunctionPropagtorTest(unittest.TestCase):
    def testPropagation(self):
        filename = "../calculations/new_s2_z0_0.npz"
        af = AutocorrelationFunction.load(filename)


        srw_beamline = createBeamline()

        propagator = AutocorrelationFunctionPropagator(srw_beamline)
        propagator.setMaximumMode(30)

        propagator.propagate(af, slow_low_memory=True)
        af.save("afptest.npz")

        srw_beamline = createBeamline2()
        propagator2 = AutocorrelationFunctionPropagator(srw_beamline)
        propagator2.propagate(af)
        af.save("afptest2.npz")