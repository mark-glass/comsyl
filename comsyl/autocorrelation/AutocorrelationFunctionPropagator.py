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
import inspect
import pickle
import glob, os

from socket import gethostname

try:
    import mpi4py.MPI as mpi
except:
    pass

from oasys_srw.srwlib import *


from comsyl.mathcomsyl.Twoform import Twoform
from comsyl.mathcomsyl.TwoformVectors import TwoformVectorsWavefronts
from comsyl.mathcomsyl.utils import trapez2D
from comsyl.parallel.DistributionPlan import DistributionPlan
from comsyl.waveoptics.Wavefront import NumpyWavefront
from comsyl.utils.Logger import logAll
from comsyl.parallel.utils import isMaster, barrier

def propagateWavefront(srw_beamline, wavefront, rx, drx, ry, dry, rescale_x, rescale_y, i_mode, python_to_be_used="python"):
    if isMaster():
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

    s_id=str(mpi.COMM_WORLD.Get_rank())+"_"+gethostname()
    wavefront.save("./tmp/tmp%s_in"%s_id)

    parameter_lines = "rx=%f\ndrx=%f\nry=%f\ndry=%f\nrescale_x=%f\nrescale_y=%f\ns_id=\"%s\"" % (rx, drx, ry, dry, rescale_x, rescale_y, s_id)
    pickle.dump(srw_beamline, open("./tmp/tmp%s_beamline.p"%s_id,"wb"))
    lines ="""
import pickle
from oasys_srw.srwlib import *
from comsyl.waveoptics.SRWAdapter import SRWAdapter
from comsyl.waveoptics.Wavefront import NumpyWavefront, SRWWavefront

wavefront = NumpyWavefront.load("./tmp/tmp%s_in.npz"%s_id)
adapter = SRWAdapter()
wfr = adapter.SRWWavefrontFromWavefront(wavefront,
                                        rx,
                                        drx,
                                        ry,
                                        dry,rescale_x,rescale_y)
#print("Doing propagation in external call")
srw_beamline = pickle.load(open("./tmp/tmp%s_beamline.p"%s_id,"rb"))
srwl.PropagElecField(wfr, srw_beamline)

tmp = SRWWavefront(wfr).toNumpyWavefront()
wfr = None
tmp.save("./tmp/tmp%s_out" % s_id)
"""

    file = open("./tmp/tmp%s.py"%s_id, "w")
    file.writelines(parameter_lines)
    file.writelines("\ni_mode=%d\n"%i_mode) # added srio
    file.writelines(lines)
    file.close()

    os.system(python_to_be_used+" ./tmp/tmp%s.py"%s_id)

    return NumpyWavefront.load("./tmp/tmp%s_out.npz"%s_id)


def propagateWavefrontWofry(beamline, wavefront, i_mode, python_to_be_used="python", keep_temp_files=False):

    s_id=str(mpi.COMM_WORLD.Get_rank())+"_"+gethostname()

    wavefront.save("./tmp/tmp%s_in"%s_id)

    parameter_lines = "s_id=\"%s\"" % (s_id)
    pickle.dump(beamline, open("./tmp/tmp%s_beamline.p"%s_id,"wb"))
    lines ="""
import pickle
from wofry.propagator.propagator import PropagationManager
from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D
from comsyl.waveoptics.ComsylWofryBeamline import ComsylWofryBeamline

# initialize propagator
mypropagator = PropagationManager.Instance()
try:
    mypropagator.add_propagator(FresnelZoomXY2D())
except:
    print("May be you alreay initialized propagator and stored FresnelZoomXY2D")

beamline = pickle.load(open("./tmp/tmp%s_beamline.p"%s_id,"rb"))

ComsylWofryBeamline.propagate_numpy_wavefront(
    "./tmp/tmp%s_in.npz"%s_id,
    "./tmp/tmp%s_out.npz"%s_id,
    beamline,mypropagator)
"""

    file = open("./tmp/tmp%s.py"%s_id, "w")
    file.writelines(parameter_lines)
    file.writelines("\ni_mode=%d\n"%i_mode) # added srio
    file.writelines(lines)
    file.close()

    os.system(python_to_be_used+" ./tmp/tmp%s.py"%s_id)
    if keep_temp_files: # keep a copy of all files
        logAll("cp %s  %s" % ("./tmp/tmp%s_in.npz" % s_id,   "./tmp/tmp%s_in_mode%d.npz" % (s_id, i_mode)))
        logAll("cp %s  %s" % ("./tmp/tmp%s_out.npz" % s_id, "./tmp/tmp%s_out_mode%d.npz" % (s_id, i_mode)))
        os.system("cp %s  %s" % ("./tmp/tmp%s_in.npz" % s_id,  "./tmp/tmp%s_in_mode%d.npz" % (s_id, i_mode)))
        os.system("cp %s  %s" % ("./tmp/tmp%s_out.npz" % s_id, "./tmp/tmp%s_out_mode%d.npz" % (s_id, i_mode)))

    return NumpyWavefront.load("./tmp/tmp%s_out.npz"%s_id)


class AutocorrelationFunctionPropagator(object):
    def __init__(self, srw_beamline):
        self.__srw_beamline = srw_beamline # srio@esrf.eu: this is the beamline, can be SRW or WOFRY

        self.setMaximumMode(None)
        self.setMaxCoordinates(None, None, None, None)

    def setMaxCoordinates(self, x_min, x_max, y_min, y_max):
        self._hard_x_min = x_min
        self._hard_x_max = x_max
        self._hard_y_min = y_min
        self._hard_y_max = y_max

    def determineNewCoordinates(self, old_coordinates, hard_limit_min, hard_limit_max):

        minimum_stepsize = 1e6
        minimum_coordinate = 1e6
        maximum_coordinate = 0.0

        for coordinates in old_coordinates:
            minimum_stepsize = min(np.diff(coordinates).min(), minimum_stepsize)
            minimum_coordinate = min(coordinates.min(), minimum_coordinate)
            maximum_coordinate = max(coordinates.max(), maximum_coordinate)

        total_minimum_stepsize = np.array([1e6])
        total_minimum_coordinate = np.array([1e6])
        total_maximum_coordinate = np.array([0.0])
        minimum_stepsize = np.array([minimum_stepsize])
        minimum_coordinate = np.array([minimum_coordinate])
        maximum_coordinate = np.array([maximum_coordinate])

        mpi.COMM_WORLD.Allreduce(minimum_stepsize, total_minimum_stepsize, op=mpi.MIN)
        mpi.COMM_WORLD.Allreduce(minimum_coordinate, total_minimum_coordinate, op=mpi.MIN)
        mpi.COMM_WORLD.Allreduce(maximum_coordinate, total_maximum_coordinate, op=mpi.MAX)

        if hard_limit_min is None:
            limited_minimum_coordinate = total_minimum_coordinate[0]
        else:
            limited_minimum_coordinate = max(hard_limit_min, total_minimum_coordinate[0])

        if hard_limit_min is None:
            limited_maximum_coordinate = total_maximum_coordinate[0]
        else:
            limited_maximum_coordinate = max(hard_limit_min, total_maximum_coordinate[0])

        new_coordinates = np.arange(limited_minimum_coordinate,
                                    limited_maximum_coordinate,
                                    total_minimum_stepsize[0])

        print(total_minimum_coordinate[0],
                                    total_maximum_coordinate[0],
                                    total_minimum_stepsize[0])
        return new_coordinates

    def setMaximumMode(self, maximum_mode):
        self._maximum_mode = maximum_mode

    def _adjustWavefrontSize(self, wavefront):
        min_size_dims = min(wavefront.dim_x(), wavefront.dim_y())

        i = 0
        max_size = 4096
        if min_size_dims > max_size:
            j = min_size_dims
            while j > max_size:
                j /= 2
                i += 1

        new_size_x = wavefront.dim_x() / 2**i
        new_size_y = wavefront.dim_y() / 2**i

        if new_size_x != wavefront.dim_x() or new_size_y != wavefront.dim_y():
            logAll("Resized to %i %i" %(new_size_x, new_size_y))
            return wavefront.asFixedSize(new_size_x, new_size_y)
        else:
            return wavefront

    def propagate(self, autocorrelation_function, filename, method='SRW', python_to_be_used="python"):

        source_filename = autocorrelation_function._io.fromFile()

        try:
            source_uid = autocorrelation_function.info().uid()
        except:
            source_uid = "None"

        autocorrelation_function.info().logStart()

        logAll("Propagating %s (%s)" % (source_filename, source_uid))


        if self._maximum_mode is None:
            number_modes = autocorrelation_function.numberModes()
        else:
            number_modes = self._maximum_mode

        if isMaster():
            if not os.path.exists("tmp"):
                os.mkdir("tmp")

        distribution_plan = DistributionPlan(mpi.COMM_WORLD, n_rows=number_modes, n_columns=1)

        n_rank = mpi.COMM_WORLD.Get_rank()
        x_coordinates = []
        y_coordinates = []
        for i_mode in distribution_plan.localRows():

            for i in range(1):
                logAll("%i doing mode index: %i/%i (max mode index: %i)" % (n_rank, i_mode, max(distribution_plan.localRows()), number_modes-1))
                if n_rank == 0:
                    sys.stdout.flush()

                wavefront = autocorrelation_function.coherentModeAsWavefront(i_mode)
                #wavefront._e_field[np.abs(wavefront._e_field)<0.000001]=0.0

                if method == 'SRW':
                    # CHANGE THIS FOR WOFRY
                    srw_wavefront = propagateWavefront(self.__srw_beamline,
                                                      wavefront,
                                                      autocorrelation_function.SRWWavefrontRx(),
                                                      autocorrelation_function.SRWWavefrontDRx(),
                                                      autocorrelation_function.SRWWavefrontRy(),
                                                      autocorrelation_function.SRWWavefrontDRy(), 1.0, 1.0, i_mode,
                                                      python_to_be_used=python_to_be_used)
                elif method == 'WOFRY':
                    srw_wavefront = propagateWavefrontWofry(self.__srw_beamline,wavefront,i_mode,python_to_be_used=python_to_be_used)
                else:
                    raise Exception("Method not known: %s"%method)

                # norm_mode = trapez2D( np.abs(srw_wavefront.E_field_as_numpy()[0,:,:,0])**2, 1, 1)**0.5
                # if norm_mode > 1e2 or np.isnan(norm_mode):
                #     print("TRY %i AFTER PROPAGATION:" % i, i_mode,norm_mode)
                #     sys.stdout.flush()
                #else:
                #    break

            #if i==19:
            #    exit()

            # if np.any(norm_srw_wavefront > 10):
            #     exit()
            #
            # if np.any(norm_wavefront > 10):
            #     exit()

            adjusted_wavefront = self._adjustWavefrontSize(srw_wavefront)
            # norm_mode = trapez2D( np.abs(adjusted_wavefront.E_field_as_numpy()[0,:,:,0])**2, 1, 1)**0.5
            # if norm_mode > 1e2 or np.isnan(norm_mode):
            #     print("TRY %i AFTER ADJUSTMENT:" % i, i_mode,norm_mode)
            #     sys.stdout.flush()
            #     exit()

            # writes a file for every wavefront
            TwoformVectorsWavefronts.pushWavefront(filename, adjusted_wavefront, index=i_mode)
            #print("Saving wavefront %i" % i_mode)

            x_coordinates.append(adjusted_wavefront.absolute_x_coordinates().copy())
            y_coordinates.append(adjusted_wavefront.absolute_y_coordinates().copy())

        mpi.COMM_WORLD.barrier()

        # replace the wavefronts bu the propagated ones
        af = self._saveAutocorrelation(autocorrelation_function, number_modes, x_coordinates, y_coordinates, filename)

        # convert from one file per wavefront to one big array
        af.Twoform().convertToTwoformVectorsEigenvectors()
        af.info().setEndTime()

        filelist = glob.glob(filename+"*")
        for f in filelist:
            os.remove(f)

        return af


    def _totalNumberWavefronts(self, wavefronts):

        number_wavefront = np.array([len(wavefronts)])
        number_wavefronts = np.array([len(wavefronts)])

        mpi.COMM_WORLD.Allreduce(number_wavefront, number_wavefronts, op=mpi.SUM)

        return int(number_wavefronts[0])

    def _saveAutocorrelation(self, autocorrelation_function, number_modes, x_coordinates, y_coordinates, filename):

        new_coordinates_x = self.determineNewCoordinates(x_coordinates, self._hard_x_min, self._hard_x_max)
        new_coordinates_y = self.determineNewCoordinates(y_coordinates, self._hard_y_min, self._hard_y_max)

        twoform = autocorrelation_function.Twoform()
        eigenvalues = twoform.eigenvalues().copy()

        distribution_plan = DistributionPlan(mpi.COMM_WORLD, n_rows=number_modes, n_columns=len(new_coordinates_x)*len(new_coordinates_y))


        if distribution_plan.myRank() == 0:
            twoform_vectors = TwoformVectorsWavefronts(new_coordinates_x, new_coordinates_y, filename)
            diagonal_elements, tde = self._getDiagonalElements(eigenvalues, twoform_vectors, distribution_plan)

            new_twoform = Twoform(new_coordinates_x,
                                  new_coordinates_y,
                                  diagonal_elements,
                                  eigenvalues,
                                  twoform_vectors)

            autocorrelation_function._setTwoform(new_twoform)
            autocorrelation_function.info().set("propagation_seperator", self.seperator())
            autocorrelation_function.info().set("propagation", self.log())

            autocorrelation_function.save(filename)

        return autocorrelation_function

    def _getDiagonalElements(self, eigenvalues, twoform_vectors, distribution_plan):

        x_coordinates = twoform_vectors.xCoordinates()
        y_coordinates = twoform_vectors.yCoordinates()

        diagonal_elements = np.zeros((len(x_coordinates), len(y_coordinates)),
                                      dtype=np.complex128)

        print("Determining diagonal elements")
        for i_mode in range(distribution_plan.totalShape()[0]):
            logAll("Processing mode: %i" % i_mode)
            new_mode = twoform_vectors.read(i_mode).coveredInterpolation(x_coordinates, y_coordinates)

            norm_mode = trapez2D( np.abs(new_mode)**2, 1, 1)**0.5
            if np.isnan(norm_mode):
                print("SKIP:", i_mode,norm_mode)
                sys.stdout.flush()
                continue

            diagonal_elements[:, :] += eigenvalues[i_mode] * np.abs(new_mode[:, :])**2

        #total_diagonal_elements = np.zeros_like(diagonal_elements)

        #mpi.COMM_WORLD.Allreduce(diagonal_elements, total_diagonal_elements, op=mpi.SUM)

        total_diagonal_elements = diagonal_elements
        return total_diagonal_elements, diagonal_elements

    def seperator(self):
        return "-------------------------"

    def log(self):

        log_string = ""

        try:
            for i_elem, opt_elem in enumerate(self.__srw_beamline.arOpt):
                log_string += self.seperator()
                optical_element = "Optical element: %s" % type(opt_elem).__name__
                log_string += optical_element

                attributes = [m for m in inspect.getmembers(opt_elem) if not "__" in m[0]]

                log_string += str(self.__srw_beamline.arProp[i_elem])
                for attribute in zip(attributes):
                    log_string += str(attribute)

                log_string += self.seperator()
        except:
            pass

        return log_string
