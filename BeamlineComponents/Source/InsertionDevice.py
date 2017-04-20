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



from numpy import pi
import scipy.constants.codata

class InsertionDevice(object):
    def __init__(self, K_vertical, K_horizontal, period_length, periods_number):
        self._K_vertical = K_vertical
        self._K_horizontal = K_horizontal
        self._period_length = period_length
        self._periods_number = periods_number

    def periodLength(self):
        return self._period_length

    def periodNumber(self):
        return self._periods_number

    def length(self):
        return self.periodNumber() * self.periodLength()

    def K_vertical(self):
        return self._K_vertical

    def K_horizontal(self):
        return self._K_horizontal

    def _magneticFieldStrengthFromK(self, K):
        codata = scipy.constants.codata.physical_constants
        speed_of_light = codata["speed of light in vacuum"][0]
        mass_electron = codata["electron mass"][0]
        elementary_charge=codata["elementary charge"][0]

        B = K * 2 * pi * mass_electron * speed_of_light / (elementary_charge * self.periodLength())

        return B

    def B_vertical(self):
        return self._magneticFieldStrengthFromK(self.K_vertical())

    def B_horizontal(self):
        return self._magneticFieldStrengthFromK(self.K_horizontal())