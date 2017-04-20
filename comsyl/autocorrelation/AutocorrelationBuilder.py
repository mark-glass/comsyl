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

from comsyl.autocorrelation.PhaseSpaceDensity import PhaseSpaceDensity
from comsyl.autocorrelation.AutocorrelationBuilderStrategies import BuilderStrategyConvolution, BuilderStrategyPython

from comsyl.utils.Logger import log

class AutocorrelationBuilder(object):
    def __init__(self, N_e, sigma_matrix, weighted_fields, x_coordinates, y_coordinates, k, strategy=None):
        self._N_e = N_e
        self._sigma_matrix = sigma_matrix

        self._density = PhaseSpaceDensity(sigma_matrix, k)
        self._weighted_fields = weighted_fields.copy()

        self._field_x_coordinates = x_coordinates.copy()
        self._field_y_coordinates = y_coordinates.copy()
        self._x_coordinates = x_coordinates.copy() # self._minkowskiSum(x_coordinates)
        self._y_coordinates = y_coordinates.copy() # self._minkowskiSum(y_coordinates)

        if strategy is None:
            if self._density.isAlphaZero():
                strategy = BuilderStrategyConvolution
            else:
                log("Found alpha not equal to zero. Can not use convolutions.")
                strategy = BuilderStrategyPython

        self.setStrategy(strategy)
        self.setAllCoordinates(self._field_x_coordinates, self._field_y_coordinates)
        self.setDoNotUseConvolutions(False)

    def setStrategy(self, strategy):
        log("Setting autocorrelation strategy: %s" % str(strategy.__name__))
        self._strategy = strategy(self._x_coordinates, self._y_coordinates,
                                  self._density,
                                  self._field_x_coordinates, self._field_y_coordinates,
                                  self._weighted_fields)

    def _minkowskiSum(self, coordinates):
        delta_coordinate = coordinates[1] - coordinates[0]
        interval = delta_coordinate * coordinates.shape[0]
        mink_sum = np.linspace(-interval, interval, coordinates.shape[0] * 1)#2-1)
        return mink_sum

    def xCoordinates(self):
        return self._x_coordinates

    def yCoordinates(self):
        return self._y_coordinates

    def staticElectronDensity(self):
        return self._strategy._rho

    def setAllCoordinates(self, x_coordinates, y_coordinates):
        self._strategy.setAllCoordinates(x_coordinates, y_coordinates)

    def evaluate(self, r_1, r_2):
        return self._strategy.evaluate(r_1, r_2)

    def evaluateAllR_2(self, r_1):
        return self._strategy.evaluateAllR_2(r_1)

    def calculateIntensity(self):
        return self._strategy.calculateIntensity()

    def setDoNotUseConvolutions(self, do_not_use_convolutions):
        self._strategy.setDoNotUseConvolutions(do_not_use_convolutions)