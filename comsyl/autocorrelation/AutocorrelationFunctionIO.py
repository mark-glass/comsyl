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

from comsyl.math.TwoformVectors import TwoformVectorsWavefronts, TwoformVectorsEigenvectors
from comsyl.parallel.utils import isMaster, barrier

class AutocorrelationFunctionIO(object):
    def __init__(self):
        self._setWasFileLoaded(None)

    def save(self, filename, af):
        af.Twoform().saveVectors(filename.replace(".npz", ""))

        if isMaster():
            print("Saving autocorrelation function to %s" % filename)
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

            print("Saving done.")

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

        filename_npz = filename.replace(".npz", "")+".npz"
        filename_data = filename.replace(".npz", "")+".npy"

        try:
            #vectors = TwoformVectorsEigenvectors(np.load(filename_data, mmap_mode="r"))
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