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

from comsyl.utils.Logger import log
from comsyl.waveoptics.SRWAdapter import SRWAdapter

VIRTUAL_SOURCE_CENTER = "center"
VIRTUAL_SOURCE_ENTRANCE = "entrance"

class WavefrontBuilder(object):
    def __init__(self, undulator, sampling_factor, min_dimension_x, max_dimension_x, min_dimension_y, max_dimension_y, energy, source_position):
        self._undulator = undulator
        self._sampling_factor = sampling_factor
        self._min_dimension_x = min_dimension_x
        self._max_dimension_x = max_dimension_x
        self._min_dimension_y = min_dimension_y
        self._max_dimension_y = max_dimension_y
        self._photon_energy = energy
        self._source_position = source_position

    def _applyLimits(self, value, minimum, maximum):
        if minimum > value:
            return minimum
        elif maximum < value:
            return maximum
        else:
            return value

    def _setAdapterInitialZ(self, adapter):
        if self._source_position == VIRTUAL_SOURCE_CENTER:
            adapter._initial_z = 0.0
            raise NotImplementedError("CENTER position might need correction. Not yet implemented.")
        elif self._source_position == VIRTUAL_SOURCE_ENTRANCE:
            adapter._initial_z = self._undulator.period_length()*3 + self._undulator.length() / 2.0
        else:
            raise NotImplementedError("Source position %s" % self._source_position)

    def build(self, electron_beam, xp, yp, z_offset,x=0.0, y=0.0, initial_z=None):
        adapter = SRWAdapter()
        adapter.setSamplingFactor(self._sampling_factor)

        max_theta_x = self._undulator.gaussian_central_cone_aperture(electron_beam.gamma(),n=1) * 3.0
        z = self._undulator.length() + z_offset

        min_dimension_x_theta = 1.0 * self._min_dimension_x / z * np.sqrt(2.0)
        max_dimension_x_theta = 1.0 * self._max_dimension_x / z * np.sqrt(2.0)

        min_dimension_y_theta = 1.0 * self._min_dimension_y / z * np.sqrt(2.0)
        max_dimension_y_theta = 1.0 * self._max_dimension_y / z * np.sqrt(2.0)

        max_theta_x = self._applyLimits(max_theta_x, min_dimension_x_theta, max_dimension_x_theta)

        max_theta_y = self._applyLimits(max_theta_x / 1.5, min_dimension_y_theta, max_dimension_y_theta)

        self._setAdapterInitialZ(adapter)
        log("Using initial z_0 for initial conditions: %e" % adapter._initial_z)

        calc_wavefront = adapter.wavefrontRectangularForSingleEnergy(electron_beam,
                                                                     self._undulator, z,
                                                                     max_theta_x,
                                                                     max_theta_y,
                                                                     self._photon_energy,
                                                                     x=x,
                                                                     xp=xp,
                                                                     y=y,
                                                                     yp=yp)
        return calc_wavefront

    def buildOnGrid(self, reference_wavefront, electron_beam, z_offset, xp, yp, x=0.0, y=0.0, ):
        adapter = SRWAdapter()
        adapter.setSamplingFactor(self._sampling_factor)

        z = self._undulator.length() + z_offset

        grid_length_x = reference_wavefront.absolute_x_coordinates().max()
        grid_length_y = reference_wavefront.absolute_y_coordinates().max()

        energy = self._photon_energy

        self._setAdapterInitialZ(adapter)
        log("Using initial z_0 for initial conditions: %e" % adapter._initial_z)

        calc_wavefront = adapter.wavefrontByCoordinates(electron_beam=electron_beam,
                                                        undulator=self._undulator,
                                                        z_start=z,
                                                        grid_length_x=grid_length_x,
                                                        grid_length_y=grid_length_y ,
                                                        energy_number=1, energy_start=energy, energy_end=energy,
                                                        x=x,
                                                        xp=xp,
                                                        y=y,
                                                        yp=yp)
        return calc_wavefront

    def createReferenceWavefrontAtVirtualSource(self, Rx, dRx, Ry, dRy, configuration, source_position, wavefront):
        adapter = SRWAdapter()

        if source_position == VIRTUAL_SOURCE_CENTER:
            z = -1.0 * self._undulator.length()
        elif source_position == VIRTUAL_SOURCE_ENTRANCE:
            z = -1.5 * self._undulator.length() #- 2 * self._undulator.periodLength()
        else:
            raise NotImplementedError("Source position %s" % source_position)

        log("Using source position: %s with z=%.02f" % (source_position, z))

        wavefront = adapter.propagate(wavefront, Rx, dRx, Ry, dRy, z)

        x_min = -configuration.sourceWavefrontMaximalSizeHorizontal()
        x_max = -x_min
        y_min = -configuration.sourceWavefrontMaximalSizeVertical()
        y_max = -y_min

        if x_min > wavefront.minimal_x_coodinate() or x_max < wavefront.maximal_x_coodinate() or \
           y_min > wavefront.minimal_y_coodinate() or y_max < wavefront.maximal_y_coodinate():

            dim_x = int((x_max-x_min)/wavefront.x_stepwidth())
            dim_y = int((y_max-y_min)/wavefront.y_stepwidth())

            divisor_x = configuration.samplingFactorDivisorHorizontal()
            if divisor_x == "":
                divisor_x = 1.0

            divisor_y = configuration.samplingFactorDivisorVertical()
            if divisor_y == "":
                divisor_y = 1.0


            wavefront = wavefront.onDomain(x_min, x_max, int(dim_x/divisor_x),
                                           y_min, y_max, int(dim_y/divisor_y))
        #wavefront = wavefront.zeroPadded(zero_padding_factor_x=1.0, zero_padding_factor_y=1.0)
        return wavefront
