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
try:
    import mpi4py.MPI as mpi
except:
    pass
from comsyl.parallel.utils import isMaster, barrier

from oasys_srw.srwlib import *

from syned.storage_ring.electron_beam import ElectronBeam
from comsyl.autocorrelation.AutocorrelationBuilder import AutocorrelationBuilder
from comsyl.autocorrelation.AutocorrelationFunction import AutocorrelationFunction
from comsyl.autocorrelation.AutocorrelationInfo import AutocorrelationInfo
from comsyl.autocorrelation.AutocorrelationOperator import AutocorrelationOperator
from comsyl.autocorrelation.DivergenceAction import DivergenceAction
from comsyl.mathcomsyl.Twoform import Twoform
from comsyl.mathcomsyl.MatrixBuilder import MatrixBuilder
from comsyl.mathcomsyl.Eigenmoder import Eigenmoder
from comsyl.mathcomsyl.EigenmoderSeparation import EigenmoderSeparation
from comsyl.infos.Undulators import undulatorByName
from comsyl.infos.Lattices import latticeByName
from comsyl.autocorrelation.PhaseSpaceDensity import PhaseSpaceDensity
from comsyl.utils.Logger import log, resetLog

from comsyl.waveoptics.WavefrontBuilder import WavefrontBuilder, VIRTUAL_SOURCE_CENTER, VIRTUAL_SOURCE_ENTRANCE
from comsyl.waveoptics.GaussianWavefrontBuilder import GaussianWavefrontBuilder
from comsyl.autocorrelation.SigmaMatrix import SigmaMatrixFromCovariance


class AutocorrelationSimulator(object):
    def __init__(self):
        self._adjust_memory_consumption = False

    def _setConfiguration(self, configuration):
        self._configuration = configuration

    def configuration(self):
        return self._configuration

    def setAdjustMemoryConsumption(self, adjust_consumption, min_size_in_gb, max_size_in_gb):
        self._adjust_memory_consumption = adjust_consumption
        self._adjust_memory_min_size = min_size_in_gb
        self._adjust_memory_max_size = max_size_in_gb

    def adjustMemoryConsumption(self):
        return self._adjust_memory_consumption

    def calculateAutocorrelationForEnergy(self, wavefront, weighted_fields, sigma_matrix):

        wavenumber = wavefront.wavenumbers()[0]
        configuration = self.configuration()

        if configuration.doNotUseConvolutions() != "true":
            do_not_use_convolutions = False
        else:
            do_not_use_convolutions = True

        if configuration.useTwoStepMethod() != "true":
            af = AutocorrelationBuilder(N_e=1.0,
                                        sigma_matrix=sigma_matrix,
                                        weighted_fields=weighted_fields,
                                        x_coordinates=wavefront.absolute_x_coordinates(),
                                        y_coordinates=wavefront.absolute_y_coordinates(),
                                        k=wavenumber)


            x_coords = af._field_x_coordinates
            y_coords = af._field_y_coordinates

            if configuration.independentDimensions() == "true":
                cut_x = af._strategy.evaluateCutX()
                cut_y = af._strategy.evaluateCutY()

                eigenmoder = EigenmoderSeparation(x_coords, y_coords, cut_x, cut_y)
                twoform = eigenmoder.twoform(number_eigenvalues=configuration.numberModes())
                return af.staticElectronDensity(), twoform

            matrix_builder = MatrixBuilder(x_coords, y_coords)
            work_matrix = matrix_builder._createParallelMatrix(af.evaluateAllR_2)
        else:
            if configuration.numberModesFirstStep() == "":
                number_modes_first_step = configuration.numberModes()
            else:
                number_modes_first_step = int(configuration.numberModesFirstStep())


            ao = AutocorrelationOperator(N_e=1.0,
                                         sigma_matrix=sigma_matrix,
                                         weighted_fields=weighted_fields,
                                         x_coordinates=wavefront.absolute_x_coordinates(),
                                         y_coordinates=wavefront.absolute_y_coordinates(),
                                         k=wavenumber,
                                         number_modes=number_modes_first_step)
            work_matrix = ao
            af = ao._builder
            af.setDoNotUseConvolutions(do_not_use_convolutions)

        barrier()

        return af.staticElectronDensity(), work_matrix

    def _determineBeamEnergies(self, electron_beam, sigma_matrix, number_energies):
        e_0 = electron_beam.energy()
        sigma_dd = sigma_matrix.element("d", "d")
        if sigma_dd < 1e-60 or number_energies <= 1:
            sigma_e = 1.0
            beam_energies = np.array([0.0])
        else:
            energy_spread = sigma_dd ** -0.5
            sigma_e = e_0 * energy_spread
            beam_energies = np.linspace(-3 * sigma_e, 3 * sigma_e, number_energies)

        return e_0, sigma_e, beam_energies

    def _estimateMemoryConsumption(self, wavefront):
        configuration = self.configuration()

        size_x = wavefront.absolute_x_coordinates().size
        size_y = wavefront.absolute_y_coordinates().size
        size_wavefront = size_x * size_y
        size_matrix = size_wavefront ** 2
        size_matrix_per_core = size_matrix / mpi.COMM_WORLD.size
        size_modes = size_wavefront * configuration.numberModes()
        size_solver_worst = size_wavefront * configuration.numberModes() * 12
        size_solver_worst_per_core = size_solver_worst / mpi.COMM_WORLD.size

        if isMaster():
            log("Using sampling factor: %.02f" % configuration.samplingFactor())
            log("Using %i x %i points per plane" % (size_x, size_y))
            log("A single wavefront needs %.2f mb" % (size_wavefront * 16 / 1024.**2))
            log("The total matrix needs %.2f gb" % (size_matrix * 16 / 1024.**3))
            log("Per core the matrix needs %.2f gb" % (size_matrix_per_core * 16 / 1024.**3))
            log("The modes will take %.2f mb" % (size_modes * 16 / 1024.**2))
            log("The solver needs in the worst case about %.2f gb" % (size_solver_worst * 16 / 1024.**3))
            log("Per core the solver needs in the worst case about %.2f gb" % (size_solver_worst_per_core * 16 / 1024.**3))

        return size_matrix

    def _performMemoryConsumptionAdjustment(self, sigma_matrix, undulator, info, size_matrix):
        if 16*size_matrix < self._adjust_memory_min_size * 1024.**3:
            sampling_factor = self._configuration.samplingFactor() + 0.1
            self._configuration.setSamplingFactor(sampling_factor)
            self.calculateAutocorrelation(sigma_matrix,undulator, info)

        if 16*size_matrix > self._adjust_memory_max_size * 1024.**3:
            sampling_factor = self._configuration.samplingFactor() - 0.1
            self._configuration.setSamplingFactor(sampling_factor)
            self.calculateAutocorrelation(sigma_matrix,undulator, info)

    def calculateAutocorrelation(self, electron_beam, undulator, info):

        configuration = self.configuration()

        # electron_beam = ElectronBeam(energy_in_GeV=6.04,
        #                              energy_spread=0,
        #                              average_current=0.2000,
        #                              electrons=1)

        sigma_matrix = SigmaMatrixFromCovariance(xx   = electron_beam._moment_xx   ,
                                                 xxp  = electron_beam._moment_xxp  ,
                                                 xpxp = electron_beam._moment_xpxp ,
                                                 yy   = electron_beam._moment_yy   ,
                                                 yyp  = electron_beam._moment_yyp  ,
                                                 ypyp = electron_beam._moment_ypyp ,
                                                 sigma_dd = electron_beam._energy_spread,
                                                 )

        resonance_energy = int(undulator.resonance_energy(electron_beam.gamma(), 0, 0))
        energy = resonance_energy*configuration.detuningParameter()

        if configuration.virtualSourcePosition() == "":
            if sigma_matrix.isAlphaZero():
                source_position = VIRTUAL_SOURCE_CENTER
            else:
                source_position = VIRTUAL_SOURCE_ENTRANCE
        else:
            source_position = configuration.virtualSourcePosition()

        wavefront_builder = WavefrontBuilder(undulator=undulator,
                                             sampling_factor=configuration.samplingFactor(),
                                             min_dimension_x=configuration.exitSlitWavefrontMinimalSizeHorizontal(),
                                             max_dimension_x=configuration.exitSlitWavefrontMaximalSizeHorizontal(),
                                             min_dimension_y=configuration.exitSlitWavefrontMinimalSizeVertical(),
                                             max_dimension_y=configuration.exitSlitWavefrontMaximalSizeVertical(),
                                             energy=energy,
                                             source_position=source_position)

        # from comsyl.waveoptics.WavefrontBuilderPySRU import WavefrontBuilderPySRU
        # wavefront_builder = WavefrontBuilderPySRU(undulator=undulator,
        #                                      sampling_factor=configuration.samplingFactor(),
        #                                      min_dimension_x=configuration.exitSlitWavefrontMinimalSizeHorizontal(),
        #                                      max_dimension_x=configuration.exitSlitWavefrontMaximalSizeHorizontal(),
        #                                      min_dimension_y=configuration.exitSlitWavefrontMinimalSizeVertical(),
        #                                      max_dimension_y=configuration.exitSlitWavefrontMaximalSizeVertical(),
        #                                      energy=energy)

        info.setSourcePosition(source_position)

        e_0, sigma_e, beam_energies = self._determineBeamEnergies(electron_beam, sigma_matrix, configuration.beamEnergies())

        # determineZ(electron_beam, wavefront_builder, sigma_matrix.sigma_x(), sigma_matrix.sigma_x_prime(),
        #                 sigma_matrix.sigma_y(), sigma_matrix.sigma_y_prime())

        sorted_beam_energies = beam_energies[np.abs(beam_energies).argsort()]

        field_x_coordinates = None
        field_y_coordinates = None
        exit_slit_wavefront = None
        first_wavefront = None

        weighted_fields = None
        for i_beam_energy, beam_energy in enumerate(sorted_beam_energies):
            # TDOO: recheck normalization, especially for delta coupled beams.
            if len(sorted_beam_energies) > 1:
                stepwidth_beam = np.diff(beam_energies)[0]
                weight = (1/(2*np.pi*sigma_e**2)**0.5) * np.exp(-beam_energy**2/(2*sigma_e**2)) * stepwidth_beam
            else:
                weight = 1.0
            electron_beam._energy = e_0 + beam_energy


            log("%i/%i: doing energy: %e with weight %e" %(i_beam_energy+1, len(beam_energies),
                                                           electron_beam.energy(),  weight))

            # Prepare e_field
            if field_x_coordinates is None or field_y_coordinates is None:
                wavefront = wavefront_builder.build(electron_beam,
                                                    xp=0.0,
                                                    yp=0.0,
                                                    z_offset=0.0)

                reference_wavefront = wavefront.toNumpyWavefront()
            else:
                wavefront = wavefront_builder.buildOnGrid(reference_wavefront,
                                                          electron_beam,
                                                          xp=0.0,
                                                          yp=0.0,
                                                          z_offset=0.0)

            try:
                Rx, dRx, Ry, dRy = wavefront.instantRadii()
            except AttributeError:
                Rx, dRx, Ry, dRy = 0,0,0,0

            energy = wavefront.energies()[0]
            wavefront = wavefront.toNumpyWavefront()

            if exit_slit_wavefront is None:
                exit_slit_wavefront = wavefront.clone()

            wavefront = wavefront_builder.createReferenceWavefrontAtVirtualSource(Rx, dRx, Ry, dRy, configuration, source_position, wavefront)

            if self.configuration().useGaussianWavefront() == "true":
                gaussian_wavefront_builder = GaussianWavefrontBuilder()
                wavefront = gaussian_wavefront_builder.fromWavefront(wavefront, info)

            if field_x_coordinates is None or field_y_coordinates is None:
                wavefront = wavefront.asCenteredGrid(resample_x=1.0,
                                                     resample_y=1.0)
                field_x_coordinates = np.array(wavefront.absolute_x_coordinates())
                field_y_coordinates = np.array(wavefront.absolute_y_coordinates())
            else:
                wavefront = wavefront.asCenteredGrid(field_x_coordinates, field_y_coordinates)

            size_matrix = self._estimateMemoryConsumption(wavefront)

            if self.adjustMemoryConsumption():
                self._performMemoryConsumptionAdjustment(sigma_matrix, undulator, info, size_matrix)
                exit(0)

            if first_wavefront is None:
                first_wavefront = wavefront.clone()

            if weighted_fields is None:
                weighted_fields = np.zeros((len(sorted_beam_energies), len(field_x_coordinates), len(field_y_coordinates)), dtype=np.complex128)

            weighted_fields[i_beam_energy, :, :] = np.sqrt(weight) * wavefront.E_field_as_numpy()[0, :, :, 0].copy()

        log("Broadcasting electrical fields")
        weighted_fields = mpi.COMM_WORLD.bcast(weighted_fields, root=0)

        static_electron_density, work_matrix = self.calculateAutocorrelationForEnergy(wavefront,
                                                                                      weighted_fields,
                                                                                      sigma_matrix)

        electron_beam._energy = e_0

        if isinstance(work_matrix, AutocorrelationOperator):
            log("Setting up eigenmoder")
            eigenmoder = Eigenmoder(field_x_coordinates, field_y_coordinates)
            log("Starting eigenmoder")

            eigenvalues_spatial, eigenvectors_parallel = eigenmoder.eigenmodes(work_matrix, work_matrix.numberModes(), do_not_gather=True)

            if isMaster():
                total_spatial_mode_intensity = eigenvalues_spatial.sum() * work_matrix._builder._density.normalizationConstant() / np.trapz(np.trapz(work_matrix.trace(), field_y_coordinates), field_x_coordinates)
                info.setTotalSpatialModeIntensity(total_spatial_mode_intensity)
                log("Total spatial mode intensity: %e" % total_spatial_mode_intensity.real)


            if configuration.twoStepDivergenceMethod() == "":
                divergence_method = "accurate"
            else:
                divergence_method = configuration.twoStepDivergenceMethod()

            divergence_action = DivergenceAction(x_coordinates=field_x_coordinates,
                                                 y_coordinates=field_y_coordinates,
                                                 intensity=work_matrix.trace(),
                                                 eigenvalues_spatial=eigenvalues_spatial,
                                                 eigenvectors_parallel=eigenvectors_parallel,
                                                 phase_space_density=PhaseSpaceDensity(sigma_matrix, wavefront.wavenumbers()[0]),
                                                 method=divergence_method)

            twoform = divergence_action.apply(number_modes=configuration.numberModes())
        elif isinstance(work_matrix, Twoform):
            twoform = work_matrix
        else:
            eigenmoder = Eigenmoder(field_x_coordinates, field_y_coordinates)
            twoform = eigenmoder.eigenmodes(work_matrix, configuration.numberModes())

        total_beam_energies = e_0 + beam_energies

        info.setEndTime()
        autocorrelation_function = AutocorrelationFunction(sigma_matrix=sigma_matrix,
                                                           undulator=undulator,
                                                           detuning_parameter=configuration.detuningParameter(),
                                                           energy=energy,
                                                           electron_beam_energy=electron_beam.energy(),
                                                           wavefront=first_wavefront,
                                                           exit_slit_wavefront=exit_slit_wavefront,
                                                           srw_wavefront_rx=Rx,
                                                           srw_wavefront_drx=dRx,
                                                           srw_wavefront_ry=Ry,
                                                           srw_wavefront_dry=dRy,
                                                           sampling_factor=configuration.samplingFactor(),
                                                           minimal_size=configuration.exitSlitWavefrontMinimalSizeVertical(),
                                                           beam_energies=total_beam_energies,
                                                           weighted_fields=weighted_fields,
                                                           static_electron_density=static_electron_density,
                                                           twoform=twoform,
                                                           info=info)

        return autocorrelation_function

    def _determineFilename(self):
        configuration = self._configuration

        sampling_factor = str(configuration.samplingFactor())
        directory = "./calculations/"
        filename = directory + configuration.filename()
        if configuration.filenameSuffix() != "":
            filename += "_" + configuration.filenameSuffix()
        filename += "_s" + str(sampling_factor)

        return filename

    def _runConvergenceSeries(self, configuration):
        sampling_factors = np.linspace(configuration.convergenceSeriesStart(),
                                       configuration.samplingFactor(),
                                       configuration.convergenceSeriesLength())

        for i_factor, sampling_factor in enumerate(sampling_factors):
            log("Doing sampling factor %e (%i/%i)" % (sampling_factor, i_factor+1, len(sampling_factors)))
            simulation = AutocorrelationSimulator()
            current_configuration = configuration.clone()
            current_configuration.setDoConvergenceSeries(False)
            current_configuration.setSamplingFactor(str(sampling_factor))

            filename = simulation.run(current_configuration)

        return filename

    def _runSingleSimulation(self, configuration):
        resetLog()

        self._setConfiguration(configuration)
        filename = self._determineFilename()

        lattice_name = configuration.latticeName()
        electron_beam = latticeByName(lattice_name)
        undulator = undulatorByName(configuration.undulatorName())

        info = AutocorrelationInfo()
        info.logStart()
        info.setConfiguration(configuration)
        info.setTag(configuration.tag())
        af = self.calculateAutocorrelation(electron_beam=electron_beam,
                                           undulator=undulator,
                                           info=info)

        af.save(filename)
        log("File created: %s" % filename+".npy")
        log("File created: %s" % filename+".npz")
        try:
            pass
            # import h5py
            # af.saveh5(filename+".h5")
            # log("File created: %s" % filename+".h5")
        except:
            pass

        barrier()

        return filename

    def run(self, configuration):
        if configuration.doConvergenceSeries():
            return self._runConvergenceSeries(configuration)
        else:
            return self._runSingleSimulation(configuration)
