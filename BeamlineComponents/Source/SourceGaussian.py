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



"""
Represents an optical Gaussian source.
"""

class SourceGaussian(object):
    def __init__(self, sigma_x, sigma_y, sigma_x_prime, sigma_y_prime, energy):
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y
        self._sigma_x_prime = sigma_x_prime
        self._sigma_y_prime = sigma_y_prime
        self._energy = energy

    def setAveragePhotonEnergy(self, average_photon_energy):
        self.__average_photon_energy = average_photon_energy
    
    def averagePhotonEnergy(self):
        return self.__average_photon_energy

    def setPulseEnergy(self, pulse_energy):
        self.__pulse_energy = pulse_energy 
    
    def pulseEnergy(self):
        return self.__pulse_energy
    
    def setRepititionRate(self, repitition_rate):
        self.__repitition_rate = repitition_rate

    def repititionRate(self):
        return self.__repitition_rate

    def setPolarization(self, polarization):
        self.__polarization = polarization
        
    def polarization(self):
        return self.__polarization

    def sigmaX(self):
        return self._sigma_x

    def sigmaY(self):
        return self._sigma_y

    def sigmaXPrime(self):
        return self._sigma_x_prime

    def sigmaYPrime(self):
        return self._sigma_y_prime

    def energy(self):
        return self._energy
