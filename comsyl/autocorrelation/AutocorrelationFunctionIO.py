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
import sys
from comsyl.utils.Logger import log, logProgress

try:
    import h5py
    has_h5py = True
except:
    print("h5py is not installed")
    has_h5py = False


from comsyl.mathcomsyl.TwoformVectors import TwoformVectorsWavefronts, TwoformVectorsEigenvectors
from syned.storage_ring.magnetic_structures.insertion_device import InsertionDevice
from comsyl.parallel.utils import isMaster, barrier

def undulator_as_numpy_array(self):
    array = np.array([self.K_vertical(),
                      self.K_horizontal(),
                      self.period_length(),
                      self.number_of_periods()])
    return array

def undulator_from_numpy_array(array):
    return InsertionDevice(array[0], array[1], array[2], array[3])


class AutocorrelationFunctionIO(object):
    def __init__(self):
        self._setWasFileLoaded(None)

    def save(self, filename, af):
        af.Twoform().saveVectors(filename.replace(".npz", ""))

        if isMaster():
            sys.stdout.flush()

            data_dict = af.asDictionary()
            save_dict = dict()
            for key in data_dict.keys():
                if key !="twoform_4":
                    save_dict["np_"+key] = data_dict[key]
                else:
                    eigenvectors = data_dict["twoform_4"]

            filename_npz = filename.replace(".npz", "")

            np.savez_compressed(filename_npz, **save_dict)


    def saveh5(self, filename, af, maximum_number_of_modes=None):
        if has_h5py == False:
            raise ImportError("h5py not available")

        if maximum_number_of_modes is None:
            maximum_number_of_modes = af.Twoform().numberVectors() # af.numberModes()


        if maximum_number_of_modes > af.numberModes():
            maximum_number_of_modes = af.numberModes()

        file_array_shape = (maximum_number_of_modes,
                            len(af.Twoform().xCoordinates()),
                            len(af.Twoform().yCoordinates()))

        if isMaster():
            f = h5py.File(filename, 'w')
            bigdataset = f.create_dataset("twoform_4",file_array_shape, dtype=np.complex)

        for i_vector in range(maximum_number_of_modes):
            logProgress(af.Twoform().numberVectors(), i_vector, "Writing vectors")

            vector = af.Twoform().vector(i_vector)

            if isMaster():
                bigdataset[i_vector,:,:] = vector

        if isMaster():
            sys.stdout.flush()

            data_dict = af.asDictionary()

            for key in data_dict.keys():

                if (key =="twoform_4"): # alreary done
                    pass
                elif (key == "twoform_3"):
                    if (data_dict[key] is not None):
                        f[key] = (data_dict[key])[0:maximum_number_of_modes]
                else:
                    if (data_dict[key] is not None):
                        f[key] = data_dict[key]

            f.close()


    def fromFile(self):
        return self._from_file

    def _setWasFileLoaded(self, filename):
        if filename is None:
            self._was_file_loaded = False
        else:
            self._was_file_loaded = True
            self._from_file = filename

    def _wasFileLoaded(self):
        return self._was_file_loaded

    def updateFile(self, af):
        if not self._wasFileLoaded():
            raise Exception("Was not loaded from file - can not update")
        else:
            self.save(self._from_file, af)

    @staticmethod
    def load(filename):

        filename_extension = filename.split(".")[-1]

        if filename_extension == "npz":
            filename_npz = filename.replace(".npz", "")+".npz"
            filename_data = filename.replace(".npz", "")+".npy"
        elif filename_extension == "npy":
            filename_npz = filename.replace(".npy", "")+".npz"
            filename_data = filename.replace(".npy", "")+".npy"

        try:
            file_content = np.load(filename_npz)
            vectors_shape = (file_content["np_twoform_3"].size,file_content["np_twoform_0"].size,file_content["np_twoform_1"].size)
            vectors = TwoformVectorsEigenvectors(np.memmap(filename_data, dtype=np.complex128, mode='c', shape=vectors_shape))
        except:
            print("Falling back to load_npz")
            data_dict = AutocorrelationFunctionIO.load_npz(filename)

            if "twoform_4" in data_dict:
                return data_dict
            else:
                print("Loading wavefronts")
                file_content = np.load(filename_npz)
                vectors = TwoformVectorsWavefronts(file_content["np_twoform_0"],file_content["np_twoform_1"], filename)

        data_dict = dict()
        for key in file_content.keys():
            data_dict[key.replace("np_", "")] = file_content[key]

        data_dict["twoform_4"] = vectors

        return data_dict

    def loadh5(filename):

        if has_h5py == False:
            raise ImportError("h5py not available")

        try:
            h5f = h5py.File(filename,'r')
        except:
            raise Exception("Failed to read h5 file: %s"%filename)

        data_dict = dict()

        for key in h5f.keys():
            if (key !="twoform_4"):
                data_dict[key] = h5f[key].value
            else:
                data_dict[key] = TwoformVectorsEigenvectors(h5f[key].value)

        h5f.close()
        return data_dict

    @staticmethod
    def load_npz(filename):

        file_content = np.load(filename)

        data_dict = dict()
        for key in file_content.keys():
            if key != "np_twoform_4":
                data_dict[key.replace("np_", "")] = file_content[key]
            else:
                data_dict[key.replace("np_", "")] = TwoformVectorsEigenvectors(file_content[key])

        return data_dict

