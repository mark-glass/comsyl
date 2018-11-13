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

from comsyl.waveoptics.Wavefront import NumpyWavefront
from comsyl.utils.Logger import log, logProgress
from comsyl.parallel.utils import isMaster


class TwoformVectors(object):
    def __init__(self):
        pass

    def vector(self, i_vector):
        raise NotImplementedError

    def vectors(self, index_start, index_end):
        raise NotImplementedError

    def allVectors(self):
        raise NotImplementedError

    def numberVectors(self):
        raise NotImplementedError

    def shrink(self, number_vectors):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

class TwoformVectorsEigenvectors(object):
    def __init__(self, eigenvectors):
        self._eigenvectors = eigenvectors

    def vector(self, i_vector):
        return np.array(self._eigenvectors[i_vector, :, :])

    def vectors(self, index_start, index_end):
        return np.array(self._eigenvectors[index_start:index_end, :, :])

    def allVectors(self):
        #return self._eigenvectors
        return np.array(self._eigenvectors[:, :, :])

    def numberVectors(self):
        return self._eigenvectors.shape[0]

    def conjugate(self):
        self._eigenvectors = self._eigenvectors.conj()

    def shrink(self, number_vectors):
        self._eigenvectors = self._eigenvectors[0:number_vectors, :, :]

    def resize(self, min_x, max_x, min_y, max_y, x_coordinates, y_coordinates):
        x_range = (np.where((x_coordinates >= min_x) & (x_coordinates <= max_x)))[0]
        y_range = (np.where((y_coordinates >= min_y) & (y_coordinates <= max_y)))[0]

        self._eigenvectors = self._eigenvectors[:, x_range.min():x_range.max()+1, y_range.min():y_range.max()+1]

    def save(self, filename):
        file_array_shape = self._eigenvectors.shape

        if isMaster():
            fp = np.memmap(filename+".npy", dtype=np.complex128, mode='w+', shape=file_array_shape)

            for i_vector in range(self.numberVectors()):
                logProgress(self.numberVectors(), i_vector, "Writing vectors")

                vector = self.vector(i_vector)

                if isMaster():
                    fp[i_vector, :, :] = vector
            log("Flushing")
            if isMaster():
                del fp
            log("done")

class TwoformVectorsSolver(TwoformVectorsEigenvectors):
    def __init__(self, coordinates_x, coordinates_y, eigenvalues, product_eigenmodes):
        TwoformVectorsEigenvectors.__init__(self, self._productEigenmodesToEigenmodes(coordinates_x, coordinates_y, eigenvalues, product_eigenmodes))

    def _productEigenmodesToEigenmodes(self, coordinates_x, coordinates_y, eigenvalues, product_eigenmodes):

        sort_indices = np.argsort(np.abs(eigenvalues))[::-1]

        eigenmodes = np.zeros((product_eigenmodes.shape[0],
                               len(coordinates_x),
                               len(coordinates_y)),
                               dtype=np.complex128)

        for i in range(product_eigenmodes.shape[0]):
            j = sort_indices[i]
            eigenmodes[i, :, :] = product_eigenmodes[j, :].reshape(len(coordinates_x), len(coordinates_y))

        return eigenmodes

class TwoformVectorsParallelMatrix(object):
    def __init__(self, x_coordinates, y_coordinates, parallel_matrix):
        self._parallel_matrix = parallel_matrix
        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates
        self._number_vectors = self._parallel_matrix.totalShape()[0]

    def _reshapeRow(self, row):
        return np.reshape(row, (len(self._x_coordinates),
                                len(self._y_coordinates)))


    def vector(self, i_vector):
        vector_row = self._parallel_matrix.globalRow(i_vector)
        return self._reshapeRow(vector_row)

    def vectors(self, index_start, index_end):
        vectors = np.zeros((index_end-index_start+1,len(self._x_coordinates),
                            len(self._y_coordinates)), dtype=np.complex128 )

        for i in range(index_start, index_end+1):
            vectors[i, :, :] = self.vector(i)

        return vectors

    def allVectors(self):
        return self.vectors(0, self.numberVectors()-1)

    def numberVectors(self):
        return self._number_vectors

    def shrink(self, number_vectors):
        self._number_vectors = number_vectors

    def save(self, filename):
        log("Saving vectors")
        file_array_shape = (self.numberVectors(),
                            len(self._x_coordinates),
                            len(self._y_coordinates))

        if isMaster():
            fp = np.memmap(filename+".npy", dtype=np.complex128, mode='w+', shape=file_array_shape)

        for i_vector in range(self.numberVectors()):
            logProgress(self.numberVectors(), i_vector, "Writing vectors")

            vector = self.vector(i_vector)

            if isMaster():
                fp[i_vector, :, :] = vector
        log("Flushing")
        if isMaster():
            del fp
        log("done")

class TwoformVectorsWavefronts(object):
    def __init__(self, x_coordinates, y_coordinates, file):
        self._wavefronts = list()
        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates
        self._file = file

    def xCoordinates(self):
        return self._x_coordinates

    def yCoordinates(self):
        return self._y_coordinates

    def resize(self, min_x, max_x, min_y, max_y):
        self._x_coordinates = self._x_coordinates[self._x_coordinates >= min_x]
        self._x_coordinates = self._x_coordinates[self._x_coordinates <= max_x]

        self._y_coordinates = self._y_coordinates[self._y_coordinates >= min_y]
        self._y_coordinates = self._y_coordinates[self._y_coordinates <= max_y]

    def read(self, index):
        #as_array = list()
        # filename = self._file.replace(".npz", "") + str(index)+".wfs.npz"
        # srio
        filename = self._file.replace(".npz", "_") + "%04d"%(index)+".wfs.npz"
        #for i in range(3):
        #    as_array.append(np.load(filename))
        #wavefront = NumpyWavefront.fromNumpyArray(as_array[0], as_array[1], as_array[2])
        wavefront = NumpyWavefront.load(filename)
        return wavefront

    def popWavefront(self):
        cur_index = len(self._wavefronts)
        print("Popping", cur_index)
        wavefront = self.read(index=cur_index)
        self._wavefronts.append(wavefront)

    def vector(self, i_vector):
        return self.vectors(i_vector, i_vector+1)[0, :, :]

    def vectors(self, index_start, index_end):
        while len(self._wavefronts) < index_end:
            self.popWavefront()

        wavefronts = self._wavefronts[index_start:index_end]

        vectors = np.zeros((len(wavefronts), len(self._x_coordinates), len(self._y_coordinates)), dtype=np.complex128)

        for i, wavefront in enumerate(wavefronts):
            print("Interpolating", index_start+i)
            vectors[i, :, :] = wavefront.coveredInterpolation(self._x_coordinates, self._y_coordinates)

        return vectors

    def popAll(self):
        try:
            while True:
                self.popWavefront()
        except:
            pass


    def allVectors(self):
        self.popAll()
        return self.vectors(0, self.numberVectors())

    def numberVectors(self):
        self.popAll()
        return len(self._wavefronts)

    def shrink(self, number_vectors):
        self._eigenvectors = self._eigenvectors[0:number_vectors, :, :]

    def close(self):
        self._file.close()

    def save(self, filename):
        pass

    @staticmethod
    def pushWavefront(file, wavefront, index):
        # filename = file.replace(".npz", ".") + str(index)+".wfs"
        # srio
        filename = file.replace(".npz", "_") + "%04d"%(index) + ".wfs"

        wavefront.toNumpyWavefront().save(filename)

        #for element in wavefront.toNumpyWavefront().asNumpyArray():
        #    np.save(filename, element)

    def convertToTwoformVectorsEigenvectors(self):
        return TwoformVectorsEigenvectors(self.allVectors())