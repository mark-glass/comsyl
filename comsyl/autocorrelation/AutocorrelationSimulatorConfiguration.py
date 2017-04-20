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



import json
import copy
from collections import OrderedDict

from comsyl.utils.Logger import log

class AutocorrelationSimulatorConfiguration(object):
    def __init__(self):
        self._setting = dict()
        self._available_settings=self._createAvailableSettings()

    def _createAvailableSettings(self):
        available = OrderedDict([
                                 ("lattice_name", "Name of the lattice to be used. See in available lattices in infos/Lattices.py"),
                                 ("undulator_name", "Name of the undulator to be used. See in available undulators in infos/Undulators.py"),
                                 ("sampling_factor", "Sampling point desnity. Higher is more accurate. Memory scales with N^4 of this parameter. It is one of the primary convergence parameters."),
                                 ("beam_energies", "Number of energies to sample energy spread. 0 means no energy spread."),
                                 ("number_modes", "The number of coherent modes requested to calculated. They will be in descending order according to their eigenvalues/intensity. For convergence the sum of eigenvalues should sum up to the total intensity."),
                                 ("resonance_detuning", "Determine at which energy to calculate the modes. 1.0 means first harmonic. 3.0 third harmonic and 0.999 would be 0.001 below the first harmonic."),
                                 ("exit_slit_wavefront/minimal_size_horizontal", "Minimal horizontal size [m] of the wavefront as close as possible to the undulator exit slit. Minimal means the wavefront will have at least this size."),
                                 ("exit_slit_wavefront/maximal_size_horizontal", "Maximal horizontal size [m] of the wavefront as close as possible to the undulator exit slit. Maximal means the wavefront will have at most this size and may have been cut down."),
                                 ("exit_slit_wavefront/minimal_size_vertical", "Minimal vertical size [m] of the wavefront at the virtual source position. Minimal means the wavefront will have at least this size."),
                                 ("exit_slit_wavefront/maximal_size_vertical", "Maximal vertical size [m] of the wavefront at the virtual source position. Maximal means the wavefront will have at most this size and may have been cut down."),
                                 ("source_wavefront/minimal_size_horizontal", "Minimal horizontal size [m] of the wavefront at the virtual source position. Minimal means the wavefront will have at least this size."),
                                 ("source_wavefront/maximal_size_horizontal", "Maximal horizontal size [m] of the wavefront at the virtual source position. Maximal means the wavefront will have at most this size and may have been cut down."),
                                 ("source_wavefront/minimal_size_vertical", "Minimal vertical size [m] of the wavefront at the virtual source position. Minimal means the wavefront will have at least this size."),
                                 ("source_wavefront/maximal_size_vertical", "Maximal vertical size [m] of the wavefront at the virtual source position. Maximal means the wavefront will have at most this size and may have been cut down."),
                                 ("optional/tag", "Tag to be added/saved"),
                                 ("optional/filename", "Specific filename. If left blank a default filename will be created."),
                                 ("optional/filename_suffix", "Adds the suffix to the filename."),
                                 ("optional/eigensolver_accuracy_schur", "Convergence accuracy for the Arnoldi iteration (Schur vectors)"),
                                 ("optional/eigensolver_accuracy_projection", "Convergence accuracy for the Arnoldi iteration (Subspace projection)"),
                                 ("optional/sampling_factor_divisor_horizontal", "Divide sampling factor for horizontal direction by this number."),
                                 ("optional/sampling_factor_divisor_vertical", "Divide sampling factor for vertical direction by this number."),
                                 ("optional/do_not_use_convolutions", "Do direct integration even if convolutions would be possible."),
                                 ("optional/use_two_step_method", "If set two step method is used. Mind to converge first step total intensity > 0.99."),
                                 ("optional/use_gaussian_wavefront", "Will replace the wavefront by a Gaussian wavefront. Only for test purposes."),
                                 ("optional/do_convergence_series", "A series of calculations with increasing sampling_factor starting from convergence_start will be performed."),
                                 ("optional/convergence_series_start", "The starting sampling_factor for the convergence series."),
                                 ("optional/convergence_series_length", "The length of the convergence series."),
                                 ("optional/number_modes_first_step","Specifies number of modes used in the first step. If absent, number_of_modes is applied."),
                                 ("optional/virtual_source_position","The virtual source position: entrance or center. By default entrance is used for alpha nonzero and center for alpha equal zero"),
                                 ("optional/two_step_divergence_method", "Sets the divergence method to be used in two step calculation. May be quick or accurate(default)"),
                                 ("optional/independent_dimensions", "Create decomposition from independent horizontal and vertical cut."),
                                 ("patch/source", "Name of a template configuration file whose settings will patched according to the patch section."),
                                ])

        return available

    def clone(self):
        cloned_configuration = AutocorrelationSimulatorConfiguration()
        cloned_configuration._setting = copy.deepcopy(self._setting)
        return cloned_configuration

    def _checkAvailabilty(self, name):
        if not name.replace("patch/", "") in self._available_settings and name != "patch/source":
            raise Exception("Invalid setting: %s" % name)

    def setByName(self, name, value):
        self._checkAvailabilty(name)
        self._setting[name] = value

    def byName(self, name):
        self._checkAvailabilty(name)

        if ("optional" in name or "experimental" in name) and not self.isSet(name):
            return ""

        return self._setting[name]

    def isSet(self, name):
        return name in self._setting

    def _defaultValues(self):
        # These are default values for one special case! They should be converged.
        # However: depending on the lattice and undulator and detuning and and and they will be SIGNIFICANTLY different.
        defaults = OrderedDict([
                                ("lattice_name", "new"),
                                ("undulator_name", "esrf_u18_2m"),
                                ("sampling_factor", 2.0),
                                ("beam_energies", 31),
                                ("number_modes", 500),
                                ("resonance_detuning", 1.0),
                                ("exit_slit_wavefront/minimal_size_horizontal", 75e-6),
                                ("exit_slit_wavefront/maximal_size_horizontal", 75e-6),
                                ("exit_slit_wavefront/minimal_size_vertical", 75e-6),
                                ("exit_slit_wavefront/maximal_size_vertical", 75e-6),
                                ("source_wavefront/minimal_size_horizontal", 50e-6),
                                ("source_wavefront/maximal_size_horizontal", 50e-6),
                                ("source_wavefront/minimal_size_vertical", 50e-6),
                                ("source_wavefront/maximal_size_vertical", 50e-6),
                                ("optional/tag", ""),
                                ("optional/filename", ""),
                               ])

        return defaults

    def printInfo(self):
        for name, description in self._available_settings.items():
            print("Name: %s" % name)
            print(description)
            print("-----------------")

    def setSamplingFactor(self, sampling_factor):
        self.setByName("sampling_factor", sampling_factor)

    def latticeName(self):
        return self.byName("lattice_name")

    def undulatorName(self):
        return self.byName("undulator_name")

    def samplingFactor(self):
        return float(self.byName("sampling_factor"))

    def beamEnergies(self):
        return self.byName("beam_energies")

    def numberModes(self):
        return self.byName("number_modes")

    def numberModesFirstStep(self):
        return self.byName("optional/number_modes_first_step")

    def detuningParameter(self):
        return self.byName("resonance_detuning")

    def exitSlitWavefrontMinimalSizeHorizontal(self):
        return self.byName("exit_slit_wavefront/minimal_size_horizontal")

    def exitSlitWavefrontMaximalSizeHorizontal(self):
        return self.byName("exit_slit_wavefront/maximal_size_horizontal")

    def exitSlitWavefrontMinimalSizeVertical(self):
        return self.byName("exit_slit_wavefront/minimal_size_vertical")

    def exitSlitWavefrontMaximalSizeVertical(self):
        return self.byName("exit_slit_wavefront/maximal_size_vertical")

    def sourceWavefrontMinimalSizeHorizontal(self):
        return self.byName("source_wavefront/minimal_size_horizontal")

    def sourceWavefrontMaximalSizeHorizontal(self):
        return self.byName("source_wavefront/maximal_size_horizontal")

    def sourceWavefrontMinimalSizeVertical(self):
        return self.byName("source_wavefront/minimal_size_vertical")

    def sourceWavefrontMaximalSizeVertical(self):
        return self.byName("source_wavefront/maximal_size_vertical")

    def tag(self):
        return self.byName("optional/tag")

    def setFilename(self, filename):
        self.setByName("optional/filename", filename)

    def filename(self):
        return self.byName("optional/filename")

    def filenameSuffix(self):
        return self.byName("optional/filename_suffix")

    def eigensolverAccuracySchur(self):
        return self.byName("optional/eigensolver_accuracy_schur")

    def eigensolverAccuracyProjection(self):
        return self.byName("optional/eigensolver_accuracy_projection")

    def samplingFactorDivisorHorizontal(self):
        return self.byName("optional/sampling_factor_divisor_horizontal")

    def samplingFactorDivisorVertical(self):
        return self.byName("optional/sampling_factor_divisor_vertical")

    def doNotUseConvolutions(self):
        return self.byName("optional/do_not_use_convolutions")

    def useTwoStepMethod(self):
        return self.byName("optional/use_two_step_method")

    def useGaussianWavefront(self):
        return self.byName("optional/use_gaussian_wavefront")

    def setDoConvergenceSeries(self, do_convergence_series):
        self.setByName("optional/do_convergence_series", str(do_convergence_series))

    def doConvergenceSeries(self):
        res = self.byName("optional/do_convergence_series")

        if res is not None:
            if res.lower() == "true":
                res = True
            else:
                res = False

        return res

    def convergenceSeriesStart(self):
        return float(self.byName("optional/convergence_series_start"))

    def convergenceSeriesLength(self):
        return int(self.byName("optional/convergence_series_length"))

    def virtualSourcePosition(self):
        return self.byName("optional/virtual_source_position")

    def twoStepDivergenceMethod(self):
        return self.byName("optional/two_step_divergence_method")

    def independentDimensions(self):
        return self.byName("optional/independent_dimensions")

    def _dictonaryFromFullNames(self, fullnames):
        def process_defaults(defaults, json_dictionary):
            for key, value in defaults.items():
                if "/" in key:
                    sub_key = key.split("/")[0]
                    sub_value = key.split("/")[1]

                    if not sub_key in json_dictionary:
                        json_dictionary[sub_key] = dict()

                    json_dictionary[sub_key][sub_value] = value
                else:
                    json_dictionary[key] = value

        json_dictionary = OrderedDict()
        process_defaults(fullnames, json_dictionary)

        return json_dictionary

    def createDefaultJson(self, filename):
        defaults = self._defaultValues()

        json_dictionary = self._dictonaryFromFullNames(defaults)

        json.dump(json_dictionary, open(filename, "w"),
                  sort_keys=False,
                  indent=4,
                  separators=(',', ': '))

        print("Wrote defaults to %s" % filename)

    def toString(self):
        json_dictionary = self._dictonaryFromFullNames(self._setting)

        json_string = json.dumps(json_dictionary,
                                 sort_keys=False,
                                 indent=4,
                                 separators=(',', ': '))
        return json_string

    @staticmethod
    def fromJson(json_filename):
        json_content = json.load(open(json_filename, "r"))

        configuration = AutocorrelationSimulatorConfiguration()

        def process_json_content(json_content, conf, root_key):
          for key, value in json_content.items():
            if isinstance(value, dict):
              process_json_content(value, conf, root_key+key+"/")
            else:
              conf.setByName(root_key + key, value)

        process_json_content(json_content, configuration, "")

        if configuration.isSet("patch/source"):
            patch_configuration = configuration
            template_filename = configuration.byName("patch/source")
            log("Patching configuration: %s" % template_filename)
            initial_configuration = AutocorrelationSimulatorConfiguration.fromJson(template_filename)

            for patch_setting, value in patch_configuration._setting.items():
                setting_name = patch_setting.replace("patch/", "")

                if setting_name == "source":
                    continue

                log("Patching: %s = %s" %(setting_name, str(value)) )

                initial_configuration.setByName(setting_name, value)

            return initial_configuration
        else:
            return configuration

    def __eq__(self, other):
        return self._setting == other._setting

    def __ne__(self, other):
        return not (self == other)

if __name__ == "__main__":
    configuration = AutocorrelationSimulatorConfiguration()
    configuration.printInfo()
    configuration.createDefaultJson("default.json")