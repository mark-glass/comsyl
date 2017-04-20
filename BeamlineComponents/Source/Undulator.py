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
import scipy.constants.codata

from BeamlineComponents.Source.InsertionDevice import InsertionDevice

class Undulator(InsertionDevice):

    def __init__(self, K_vertical, K_horizontal, period_length, periods_number):
        InsertionDevice.__init__(self, K_vertical, K_horizontal, period_length, periods_number)

    def resonanceWavelength(self, gamma, theta_x, theta_z):
        wavelength = (self.periodLength() / (2.0*gamma **2)) * \
                     (1 + self.K_vertical()**2 / 2.0 + self.K_horizontal()**2 / 2.0 + \
                      gamma**2 * (theta_x**2 + theta_z ** 2))
        return wavelength

    def resonanceFrequency(self, gamma, theta_x, theta_z):
        codata = scipy.constants.codata.physical_constants
        codata_c = codata["speed of light in vacuum"][0]

        frequency = codata_c / self.resonanceWavelength(gamma, theta_x, theta_z)
        return frequency

    #changed to input GAMMA def resonanceEnergy(self, energy_in_GeV, theta_x, theta_y, harmonic=1):
    def resonanceEnergy(self, gamma, theta_x, theta_y, harmonic=1):
        codata = scipy.constants.codata.physical_constants
        #energy_in_ev = codata["Planck constant"][0] * self.resonanceFrequency(energy_in_GeV*1e3/codata["electron mass energy equivalent in MeV"][0], theta_x, theta_y) / codata["elementary charge"][0]
        energy_in_ev = codata["Planck constant"][0] * self.resonanceFrequency(gamma, theta_x, theta_y) / codata["elementary charge"][0]
        #print("Resonance energy (first harmonic): %f eV"%(energy_in_ev))
        #print("Resonance energy (%d harmonic): %f eV"%(harmonic,energy_in_ev))
        return energy_in_ev*harmonic

    def gaussianCentralConeDivergence(self, gamma, n=1):
        #return (1/(2.0*gamma))*sqrt((1.0/(n*self.periodNumber())) * (1.0 + self.K_horizontal()**2/2.0 + self.K_vertical()**2/2.0))
        return (1/gamma)*np.sqrt((1.0/(2.0*n*self.periodNumber())) * (1.0 + self.K_horizontal()**2/2.0 + self.K_vertical()**2/2.0))

    def ringDivergence(self, gamma, harmonic_number, ring_number):
        #return (1/(2.0*gamma))*sqrt((1.0/(n*self.periodNumber())) * (1.0 + self.K_horizontal()**2/2.0 + self.K_vertical()**2/2.0))
        if ring_number == 0: # return central cone
            return self.gaussianCentralConeDivergence(self, gamma, n=ring_number)
        return (1/gamma)*np.sqrt( ring_number/harmonic_number * (1.0 + self.K_horizontal()**2/2.0 + self.K_vertical()**2/2.0))

    def gaussianEstimateBeamSize(self, gamma, n=1):
        #TODO: check and document
        return (2.740/(4.0*np.pi))*np.sqrt(self.periodLength()*self.resonanceWavelength(gamma,0.0,0.0)/n)

    def maximalAngularFluxEnergy(self, gamma):
        #TODO What is this? Attention that input is now gamma....
        return self.resonanceEnergy(gamma,0.0,0.0)*(1.0-1.0/float(self.periodNumber()))

    def asNumpyArray(self):
        array = np.array([self.K_vertical(),
                          self.K_horizontal(),
                          self.periodLength(),
                          self.periodNumber()])

        return array

    @staticmethod
    def fromNumpyArray(array):
        return Undulator(array[0], array[1], array[2], array[3])

    def info(self, gamma=1.0):
        print("\n=======================================================")
        print("Undulator parameters (defined):")
        print("    period: %f m" %(self.periodLength()))
        print("    number of periods: %f " %(self.periodNumber()))
        print("    K vertical: %f " %(self.K_vertical()))
        print("    K horizontal: %f " %(self.K_horizontal()))
        print("\nUndulator parameters (calculated for gamma=%f):"%(gamma))
        print("    undulator length: %f m" %(self.length()))
        print("    resonance wavelength: %g m" %(self.resonanceWavelength(gamma,0.0,0.0)))
        print("    resonance frequency: %g Hz" %(self.resonanceFrequency(gamma, 0.0,0.0)))
        print("    resonance energy harmonic 1:           %f eV" %(self.resonanceEnergy(gamma, 0.0, 0.0)))
        print("                     harmonic 3:           %f eV" %(self.resonanceEnergy(gamma, 0.0, 0.0,harmonic=3)))
        print("                     harmonic 5:           %f eV" %(self.resonanceEnergy(gamma, 0.0, 0.0,harmonic=5)))
        print("    central cone width RMS harmonic 1:     %g rad" %(self.gaussianCentralConeDivergence(gamma,1)))
        print("                           harmonic 3:     %g rad" %(self.gaussianCentralConeDivergence(gamma,3)))
        print("                           harmonic 5:     %g rad" %(self.gaussianCentralConeDivergence(gamma,5)))
        print("    first ring angle harmonic 1:           %g rad" %(self.ringDivergence(gamma, 1, 1)))
        print("                     harmonic 3:           %g rad" %(self.ringDivergence(gamma, 3, 1)))
        print("                     harmonic 5:           %g rad" %(self.ringDivergence(gamma, 5, 1)))
        print("=======================================================\n")
        return None
