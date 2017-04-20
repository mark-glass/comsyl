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
Performs coherent mode decomposition for given configuration file (first argument to script).
An example configuration file can be created by executing AutocorrelationSimulatorSample.py in the autocorrelation subdirectory.
"""
import sys
import mpi4py.MPI as mpi

from comsyl.autocorrelation.AutocorrelationSimulator import AutocorrelationSimulator
from comsyl.autocorrelation.AutocorrelationSimulatorConfiguration import AutocorrelationSimulatorConfiguration

if "--adjust-memory" in sys.argv:
    adjust_memory = True
    min_size_in_gb = float(sys.argv[3])
    max_size_in_gb = float(sys.argv[4])
else:
    adjust_memory = False

choice = "no"
if mpi.COMM_WORLD.Get_size() == 1  and adjust_memory==False:
    choice = input("Do you want to run dry? Type no for normal run.\n")

for filename in sys.argv[1:]:
    print("Processing %s" % filename)
    configuration = AutocorrelationSimulatorConfiguration.fromJson(filename)

    if configuration.filename() == "" or configuration.filename() is None:
        configuration.setFilename(filename.split("/")[-1].replace(".json", ""))


    if choice != "no":
        configuration.setSamplingFactor(0.1)

    simulator = AutocorrelationSimulator()

    if adjust_memory:
        print("Memory adjustment will be performed.")
        simulator.setAdjustMemoryConsumption(adjust_memory, min_size_in_gb, max_size_in_gb)

    simulator.run(configuration)
