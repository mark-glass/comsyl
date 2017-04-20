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
from scipy.constants import speed_of_light

from comsyl.utils.Logger import log
from comsyl.waveoptics.Wavefront import NumpyWavefront
from comsyl.waveoptics.SRWAdapter import SRWAdapter
from pySRU.ElectronBeam import ElectronBeam
from pySRU.SourceUndulatorPlane import Undulator
from pySRU.Simulation import create_simulation
from pySRU.TrajectoryFactory import TRAJECTORY_METHOD_ODE
from pySRU.RadiationFactory import RADIATION_METHOD_NEAR_FIELD, RADIATION_METHOD_FARFIELD
from pySRU.SourceUndulatorPlane import SourceUndulatorPlane

VIRTUAL_SOURCE_CENTER = "center"
VIRTUAL_SOURCE_ENTRANCE = "entrance"

class WavefrontBuilderPySRU(object):
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

    def _buildForXY(self, electron_beam, x_0, y_0, xp_0, yp_0, z, X, Y):

        #TODO: X dimension equals Y dimension?
        Y = X

        beam = ElectronBeam(Electron_energy=electron_beam.energy(), I_current=electron_beam.averageCurrent())
        undulator = Undulator(K=self._undulator.K_vertical(),
                              period_length=self._undulator.periodLength(),
                              length=self._undulator.length())

        initial_conditions = SourceUndulatorPlane(undulator=undulator, electron_beam=beam, magnetic_field=None).choose_initial_contidion_automatic()

        v_z = initial_conditions[2]

        initial_conditions[0] = xp_0 * speed_of_light
        initial_conditions[1] = yp_0 * speed_of_light
        initial_conditions[2] = np.sqrt(beam.electron_speed()**2-xp_0**2-yp_0**2) * speed_of_light
        initial_conditions[3] = x_0
        initial_conditions[4] = y_0

        if self._source_position == VIRTUAL_SOURCE_CENTER:
            initial_conditions[5] = 0.0
        print("initial cond:", initial_conditions)

        simulation = create_simulation(magnetic_structure=undulator,
                                       electron_beam=beam,
                                       traj_method=TRAJECTORY_METHOD_ODE,
                                       rad_method=RADIATION_METHOD_NEAR_FIELD, #RADIATION_METHOD_FARFIELD,
                                       distance=z,
                                       X=X,
                                       Y=Y,
                                       photon_energy=self._photon_energy,
                                       initial_condition=initial_conditions)
        #simulation.trajectory.plot_3D()
        #simulation.trajectory.plot()
        #simulation.radiation.plot()

        electrical_field = simulation.radiation_fact.calculate_electrical_field(trajectory=simulation.trajectory,
                                                                                source=simulation.source,
                                                                                distance=simulation.radiation.distance,
                                                                                X_array=simulation.radiation.X,
                                                                                Y_array=simulation.radiation.Y)

        efield = electrical_field.electrical_field()[np.newaxis, :, :, :]
        efield = efield[:, :, :, 0:2]

        calc_wavefront = NumpyWavefront(e_field=efield,
                                        x_start=X.min(),
                                        x_end=X.max(),
                                        y_start=Y.min(),
                                        y_end=Y.max(),
                                        z=z,
                                        energies=np.array([self._photon_energy]),
                                        )

        #calc_wavefront.showEField()

        self._last_simulation = simulation
        self._last_initial_conditions = initial_conditions.copy()

        return calc_wavefront


    def build(self, electron_beam, xp, yp, z_offset, x=0.0, y=0.0):
        max_theta_x = self._undulator.gaussianCentralConeDivergence(electron_beam.gamma()) * 3.0
        z = self._undulator.length() + z_offset

        min_dimension_x_theta = 1.0 * self._min_dimension_x / z * np.sqrt(2.0)
        max_dimension_x_theta = 1.0 * self._max_dimension_x / z * np.sqrt(2.0)

        min_dimension_y_theta = 1.0 * self._min_dimension_y / z * np.sqrt(2.0)
        max_dimension_y_theta = 1.0 * self._max_dimension_y / z * np.sqrt(2.0)

        max_theta_x = self._applyLimits(max_theta_x, min_dimension_x_theta, max_dimension_x_theta)

        max_theta_y = self._applyLimits(max_theta_x / 1.5, min_dimension_y_theta, max_dimension_y_theta)


        a = z * max_theta_x
        b = z * max_theta_y

        X = np.linspace(-a, a, 110 * self._sampling_factor)
        Y = np.linspace(-b, b, 110 * self._sampling_factor)

        calc_wavefront = self._buildForXY(electron_beam, x, y, xp, yp, z, X, Y)

        return calc_wavefront

    def buildOnGrid(self, reference_wavefront, electron_beam, z_offset, xp, yp, x=0.0, y=0.0):

        z = self._undulator.length() + z_offset

        calc_wavefront = self._buildForXY(electron_beam, x, y, xp, yp, z,
                                          X=reference_wavefront.absolute_x_coordinates(),
                                          Y=reference_wavefront.absolute_y_coordinates())
        return calc_wavefront

    def createReferenceWavefrontAtVirtualSource(self, Rx, dRx, Ry, dRy, configuration, source_position, wavefront):
        adapter = SRWAdapter()

        if source_position == VIRTUAL_SOURCE_CENTER:
            z = -1.0 * self._undulator.length()
        elif source_position == VIRTUAL_SOURCE_ENTRANCE:
            z = -2.0 * self._undulator.length()
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
        return wavefront
