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



import numpy

class ProbabiltyDistribution(object):
    def __init__(self, probability_density):
        self._probability_density = probability_density

    def density(self):
        return self._probability_density

    def sample(self):
        raise Exception("Must override.")

class DiscreteDistribution(ProbabiltyDistribution):
    def __init__(self, values, probabilties):
        self._values = numpy.array(values)
        self._probabilites  = numpy.array(probabilties)

        ProbabiltyDistribution.__init__(self, self._discrete_propability_density)

    def _discrete_propability_density(self, x):
        norm = numpy.sum(self._values)

        c = 0.0
        for value, probabilty in zip(self._values, self._probabilites):
            c = c + probabilty
            if c >= x:
                return value

        return self._values[-1]

    def sample(self):
        random_number = numpy.random.rand(1)[0]
        value = self._probability_density(random_number)
        return value


class DeltaDistribution(DiscreteDistribution):
    def __init__(self, value):
        DiscreteDistribution.__init__(self,[value],[1.0])


class NormalDistribution(ProbabiltyDistribution):
    def __init__(self, mean, sigma, norm):
        self._norm = norm
        self._mean = mean
        self._sigma = sigma

        f = lambda x: norm * (1.0/numpy.sqrt(2.0*numpy.pi * sigma**2)) * numpy.exp(-(x-mean)**2 / (2.0*sigma**2))
        ProbabiltyDistribution.__init__(self, f)

    def norm(self):
        return self._norm

    def mean(self):
        return self._mean

    def sigma(self):
        return self._sigma

    def sample(self):
        value =  self.norm() * numpy.random.normal(loc = self.mean(),scale = self.sigma())
        return value


class ElectronBeam(object):
    def __init__(self, energy_in_GeV, energy_spread, average_current, electrons):

        self._setTransverseXDistribution(DeltaDistribution(0.0))
        self._setTransverseYDistribution(DeltaDistribution(0.0))
        self._setLongitudinalZDistrubtion(DeltaDistribution(0.0))
        self._x_p = 0
        self._y_p = 0

        self._energy = energy_in_GeV
        self._energy_spread = energy_spread
        self._average_current = average_current
        self.setElectrons(electrons)

    def x(self):
        return self._sampleTransverseXDistribution()

    def y(self):
        return self._sampleTransverseYDistribution()

    def z(self):
        return self._sampleLongitudinalZDistrubtion

    def x_p(self):
        return self._x_p

    def y_p(self):
        return self._y_p

    def energy(self):
        return self._energy

    def gamma(self):
        return self.energy()/0.51099890221e-03 # Relative Energy

    def averageCurrent(self):
        return self._average_current

    def setElectrons(self, electrons):
        self._electrons = electrons

    def electrons(self):
        return self._electrons

    def _setTransverseXDistribution(self, transverse_x_distribution):
        self._transverse_x_distribution = transverse_x_distribution

    def _setTransverseYDistribution(self, transverse_y_distribution):
        self._transverse_y_distribution = transverse_y_distribution

    def _setLongitudinalZDistrubtion(self, longitudinal_z_distribution):
        self._longitudinal_z_distribution = longitudinal_z_distribution


    def _sampleTransverseXDistribution(self):
        return self._transverse_x_distribution.sample()

    def _sampleTransverseYDistribution(self):
        return self._transverse_y_distribution.sample()

    def _sampleLongitudinalZDistrubtion(self):
        return self._longitudinal_z_distribution.sample()