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

from comsyl.mathcomsyl.TwoformVectors import TwoformVectorsEigenvectors
from comsyl.mathcomsyl.utils import trapez2D
from comsyl.mathcomsyl.Interpolation import coveredInterpolation

# Saves the eigenvalues and the eigenvectors
class Twoform(object):
    def __init__(self, coordinates_x, coordinates_y, diagonal_elements, eigenvalues1, twoform_vectors):
        self._coordinates_x = coordinates_x
        self._coordinates_y = coordinates_y
        self._setTwoformVectors(twoform_vectors)

        eigenvalues = np.array(eigenvalues1) # added srio to deal with hdf5 files
        self._eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]

        if diagonal_elements is not None:
            self._intensity = diagonal_elements.reshape(len(self._coordinates_x), len(self._coordinates_y))
        else:
            self._intensity = self.intensityFromVectors()

        self._trace = np.sum(self._intensity)


        self.setEigenvectorErrors(np.zeros((0, 0, 0)))


        self._cached_i_x_1 = -1
        self._cached_i_y_1 = -1
        self._cached_i_y_2 = -1
        self._evaluate_cache = None
        self._evaluated_intensity = None

    def _setTwoformVectors(self, twoform_vectors):
        self._twoform_vectors = twoform_vectors

        if hasattr(self._twoform_vectors, "xCoordinates"):
            self._coordinates_x = twoform_vectors.xCoordinates()

        if hasattr(self._twoform_vectors, "yCoordinates"):
            self._coordinates_y = twoform_vectors.yCoordinates()

    def xCoordinates(self):
        if hasattr(self._twoform_vectors, "xCoordinates"):
            return self._twoform_vectors.xCoordinates()

        return self._coordinates_x

    def yCoordinates(self):
        if hasattr(self._twoform_vectors, "yCoordinates"):
            return self._twoform_vectors.yCoordinates()

        return self._coordinates_y

    def gridSpacing(self):
        dx = np.diff(self.xCoordinates())[0]
        dy = np.diff(self.yCoordinates())[0]

        return np.array([dx, dy])

    def xIndexByCoordinate(self, x):
        return np.abs(self._coordinates_x-x).argmin()

    def yIndexByCoordinate(self, y):
        return np.abs(self._coordinates_y-y).argmin()

    def intensity(self):
        return self._intensity

    def intensityFromVectors(self):
        intensity = np.zeros_like(self.vector(0))
        for i_e, eigenvalue in enumerate(self._eigenvalues):
            intensity += eigenvalue * abs(self.vector(i_e))**2

        return intensity

    def trace(self):
        return self._trace

    def numberVectors(self):
        return self._twoform_vectors.numberVectors()

    def eigenvalues(self):
        return self._eigenvalues

    def eigenvectors(self):
        return self.allVectors()

    def vector(self, i_vector):
        return self._twoform_vectors.vector(i_vector)

    def vectors(self, index_start, index_end):
        return self._twoform_vectors.vectors(index_start, index_end)

    def allVectors(self):
        return self._twoform_vectors.allVectors()

    def evaluate(self, r_1, r_2):
        i_x_1 = self.xIndexByCoordinate(r_1[0])
        i_y_1 = self.yIndexByCoordinate(r_1[1])

        i_x_2 = self.xIndexByCoordinate(r_2[0])
        i_y_2 = self.yIndexByCoordinate(r_2[1])


        res = 0.0
        eigenvectors = self.allVectors()
        # for i_ev in range(self.numberVectors()):
        #     eigenvector = eigenvectors[i_ev, :, :]
        #     res += self._eigenvalues[i_ev] * eigenvector[i_x_1, i_y_1].conj() * eigenvector[i_x_2, i_y_2]
        res = np.sum(self._eigenvalues[:] * eigenvectors[:, i_x_1, i_y_1].conj() * eigenvectors[:, i_x_2, i_y_2])

        return res

    def cachedEvaluate(self, r_1, r_2):
        i_x_1 = self.xIndexByCoordinate(r_1[0])
        i_y_1 = self.yIndexByCoordinate(r_1[1])

        if self._evaluate_cache is None:
            self._evaluate_cache = np.zeros((len(self.xCoordinates()), len(self.yCoordinates())), dtype=np.complex128)


        if i_x_1 != self._cached_i_x_1 or i_y_1 != self._cached_i_y_1:
            self._evaluate_cache[:, :] = 0.0
            print("Caching evaluate")

            try:
                self._twoform_vectors.popAll()
            except:
                pass

            for i_ev in range(self.numberVectors()):
                eigenvector = self.vector(i_ev)
                self._evaluate_cache += self._eigenvalues[i_ev] * eigenvector[i_x_1, i_y_1].conj() * eigenvector

            self._cached_i_x_1 = i_x_1
            self._cached_i_y_1 = i_y_1

        i_x_2 = self.xIndexByCoordinate(r_2[0])
        i_y_2 = self.yIndexByCoordinate(r_2[1])

        return self._evaluate_cache[i_x_2, i_y_2]

    def cachedYEvaluate(self, r_1, r_2):

        if r_1[0] > self._coordinates_x.max() or r_1[0] < self._coordinates_x.min():
            return 0.0
        if r_2[0] > self._coordinates_x.max() or r_2[0] < self._coordinates_x.min():
            return 0.0
        if r_1[1] > self._coordinates_y.max() or r_1[1] < self._coordinates_y.min():
            return 0.0
        if r_2[1] > self._coordinates_y.max() or r_2[1] < self._coordinates_y.min():
            return 0.0


        i_y_1 = self.yIndexByCoordinate(r_1[1])
        i_y_2 = self.yIndexByCoordinate(r_2[1])

        if self._evaluate_cache is None:
            self._evaluate_cache = np.zeros((len(self.xCoordinates()), len(self.xCoordinates())), dtype=np.complex128)

        if i_y_1 != self._cached_i_y_1 or i_y_2 != self._cached_i_y_2:
            self._evaluate_cache[:, :] = 0.0
            print("Caching evaluate y",r_1[1],r_2[1])

            try:
                self._twoform_vectors.popAll()
            except:
                pass

            for i_ev in range(self.numberVectors()):
                eigenvector = self.vector(i_ev)
                c_eigenvector_y1 = self._eigenvalues[i_ev]*eigenvector[:, i_y_1].conj()
                eigenvector_y2 = eigenvector[:, i_y_2]
                self._evaluate_cache[:, :] += np.outer(c_eigenvector_y1, eigenvector_y2)

            self._cached_i_y_1 = i_y_1
            self._cached_i_y_2 = i_y_2

        i_x_1 = self.xIndexByCoordinate(r_1[0])
        i_x_2 = self.xIndexByCoordinate(r_2[0])

        return self._evaluate_cache[i_x_1, i_x_2]

    def evaluateAllForFixedR1(self, x_coordinates, y_coordinates, r_1):
        values = np.zeros((x_coordinates.shape[0], y_coordinates.shape[0]), dtype=np.complex128)
        for i_x, x in enumerate(x_coordinates):
            print("Doing %i/%i" % (i_x, len(x_coordinates)))
            for i_y, y in enumerate(y_coordinates):
                r_2 = np.array([x, y])
                values[i_x, i_y] = self.cachedEvaluate(r_1, r_2)

        return values

    def evaluateInversionCut(self):

        x_indices = np.array([self.xIndexByCoordinate(x) for x in self.xCoordinates()])
        y_indices = np.array([self.yIndexByCoordinate(y) for y in self.yCoordinates()])

        x_inverse_indices = np.array([self.xIndexByCoordinate(-x) for x in self.xCoordinates()])
        y_inverse_indices = np.array([self.yIndexByCoordinate(-y) for y in self.yCoordinates()])

        res = np.zeros((len(self.xCoordinates()), len(self.yCoordinates())), dtype=np.complex128)

        eigenvalues = self.eigenvalues()

        for i_e in range(self.numberVectors()):
            vector = self.vector(i_e)

            k = (vector[x_inverse_indices, :])
            k = k[:, y_inverse_indices]
            j = vector[x_indices, :].conj()
            j = j[:, y_indices]

            res[:, :] += eigenvalues[i_e] * j[:, :] * k[:,:]

        return res

    def evaluateVCut(self):

        x_indices = np.array([self.xIndexByCoordinate(x) for x in self.xCoordinates()])
        y_indices = np.array([self.yIndexByCoordinate(y) for y in self.yCoordinates()])

        x_inverse_indices = np.array([self.xIndexByCoordinate(-x) for x in self.xCoordinates()])

        res = np.zeros((len(self.xCoordinates()), len(self.yCoordinates())), dtype=np.complex128)

        eigenvalues = self.eigenvalues()

        for i_e in range(self.numberVectors()):
            vector = self.vector(i_e)

            k = (vector[x_inverse_indices, :])
            j = vector[x_indices, :].conj()
            j = j[:, y_indices]

            res[:, :] += eigenvalues[i_e] * j[:, :] * k[:,:]

        return res

    def evaluateSmallerCut(self):

        x_indices = np.array([self.xIndexByCoordinate(x) for x in self.xCoordinates()])
        y_indices = np.array([self.yIndexByCoordinate(y) for y in self.yCoordinates()])

        y_inverse_indices = np.array([self.yIndexByCoordinate(-y) for y in self.yCoordinates()])

        res = np.zeros((len(self.xCoordinates()), len(self.yCoordinates())), dtype=np.complex128)

        eigenvalues = self.eigenvalues()

        for i_e in range(self.numberVectors()):
            vector = self.vector(i_e)

            k = vector[:, y_inverse_indices]
            j = vector[x_indices, :].conj()
            j = j[:, y_indices]

            res[:, :] += eigenvalues[i_e] * j[:, :] * k[:,:]

        return res

    def evaluateIntensity(self, r):
        if self._evaluated_intensity is None:
            self._evaluated_intensity = np.zeros((len(self.xCoordinates()), len(self.yCoordinates())))

            try:
                self._twoform_vectors.popAll()
            except:
                pass

            for i_ev in range(self.numberVectors()):
                eigenvector = self.vector(i_ev)
                self._evaluated_intensity += self._eigenvalues[i_ev].real * np.abs(eigenvector[:, :])**2

        i_x = self.xIndexByCoordinate(r[0])
        i_y = self.yIndexByCoordinate(r[1])

        return self._evaluated_intensity[i_x, i_y]

    def shrink(self, number_modes):
        self._twoform_vectors.shrink(number_modes)
        self._eigenvalues = self._eigenvalues[0:number_modes]

    def verifyTwoform(self, f, accuracy = 1e-8):

        has_inequality = False

        for i_x,x_1 in enumerate(self.xCoordinates()):
            for i_y,y_1 in enumerate(self.yCoordinates()):
                for i_x_2,x_2 in enumerate(self.xCoordinates()):
                    for i_y_2,y_2 in enumerate(self.yCoordinates()):

                        r1 = np.array([x_1, y_1])
                        r2 = np.array([x_2, y_2])

                        res_self = self.evaluate(r1, r2)
                        res_f = f(r1, r2)

                        abs_error = abs(res_f-res_self)
                        rel_error = abs(abs_error/res_f)
                        if(rel_error > accuracy and abs_error > accuracy):
                            print(rel_error,res_f,res_self, r1, r2)
                            has_inequality = True

        return not has_inequality

    def setEigenvectorErrors(self, eigenvector_errors):
        self._eigenvector_errors = eigenvector_errors.copy()

    def dot(self, vector):
        # TODO: Check scaling!
        dx = self.xCoordinates()[1] - self.xCoordinates()[0]
        dy = self.yCoordinates()[1] - self.yCoordinates()[0]

        eigenvector = self.vector(0)

        result = np.zeros_like(eigenvector)
        for i_mode in range(self.numberVectors()):
            eigenvector = self.vector(i_mode)
            alpha = trapez2D(eigenvector.conj() * vector,
                             dx=dx,
                             dy=dy)
            result += alpha * self.eigenvalues()[i_mode] * eigenvector

        return result

    def asNumpyArray(self):
        return [self.xCoordinates(), self.yCoordinates(), self._intensity, self._eigenvalues, None, self._eigenvector_errors]

    def save(self, filename):
        numpy_data_array = self.asNumpyArray()
        names = ["array_data%i" % i for i in range(len(numpy_data_array))]
        array_data = {key: value for (key, value) in zip(names, numpy_data_array)}
        np.savez_compressed(filename, **array_data)

    def saveVectors(self, filename):
        self._twoform_vectors.save(filename)

    @staticmethod
    def load(filename):
        file_content = np.load(filename)
        if not "None" in str(file_content["array_data4"]):
            twoform_vectors = TwoformVectorsEigenvectors(file_content["array_data4"])
        else:
            print("Trying loading ", filename.replace(".npz",".npy"))
            twoform_vectors = TwoformVectorsEigenvectors(np.load(filename.replace(".npz",".npy"), mmap_mode="r"))

        two_form = Twoform(file_content["array_data0"],
                           file_content["array_data1"],
                           file_content["array_data2"],
                           file_content["array_data3"],
                           twoform_vectors)

        try:
            two_form.setEigenvectorErrors(file_content["array_data5"])
        except:
            pass

        return two_form

    def resize(self, min_x, max_x, min_y, max_y):
        self._twoform_vectors.resize(min_x, max_x, min_y, max_y, self.xCoordinates(), self.yCoordinates())
        self._coordinates_x = self._coordinates_x[self._coordinates_x >= min_x]
        self._coordinates_x = self._coordinates_x[self._coordinates_x <= max_x]

        self._coordinates_y = self._coordinates_y[self._coordinates_y >= min_y]
        self._coordinates_y = self._coordinates_y[self._coordinates_y <= max_y]

        x_range = (np.where((self._coordinates_x >= min_x) & (self._coordinates_x <= max_x)))[0]
        y_range = (np.where((self._coordinates_y >= min_y) & (self._coordinates_y <= max_y)))[0]

        self._intensity = self._intensity[x_range.min():x_range.max()+1, y_range.min():y_range.max()+1]


    def convertToTwoformVectorsEigenvectors(self):
        self._twoform_vectors = self._twoform_vectors.convertToTwoformVectorsEigenvectors()
        self._eigenvalues = self.eigenvalues()[:self.numberVectors()]

    def onNewCoordinates(self, new_x_coordinates, new_y_coordinates):
        print("Resizing twoform from %ix%i to %ix%i" % (self.xCoordinates().size, self.yCoordinates().size, new_x_coordinates.size, new_y_coordinates.size))
        new_eigenvectors = np.zeros((self.numberVectors(), new_x_coordinates.size, new_y_coordinates.size), dtype=np.complex128)
        new_intensity = np.zeros((new_x_coordinates.size, new_y_coordinates.size), dtype=np.complex128)

        new_intensity[:,:] = coveredInterpolation(self.xCoordinates(), self.yCoordinates(), self.intensity().real,
                                                  new_x_coordinates, new_y_coordinates, use_uncovered=True)
        new_intensity += 1j * coveredInterpolation(self.xCoordinates(), self.yCoordinates(), self.intensity().imag,
                                                   new_x_coordinates, new_y_coordinates, use_uncovered=True)


        for i_e in range(self.numberVectors()):
            vector = self.vector(i_e)
            new_eigenvectors[i_e, :, :] = coveredInterpolation(self.xCoordinates(), self.yCoordinates(), vector.real,
                                                               new_x_coordinates, new_y_coordinates, use_uncovered=True) \
                                         + 1j*coveredInterpolation(self.xCoordinates(), self.yCoordinates(), vector.imag,
                                                                   new_x_coordinates, new_y_coordinates, use_uncovered=True)


        new_twoform_vectors = TwoformVectorsEigenvectors(new_eigenvectors)

        new_twoform = Twoform(new_x_coordinates, new_y_coordinates, new_intensity,self._eigenvalues.copy(), new_twoform_vectors)

        return new_twoform

    def XYcuts(self):
        x_coordinates = self.xCoordinates()
        y_coordinates = self.yCoordinates()
        eigenvalues = self.eigenvalues()

        x_index = np.abs(x_coordinates).argmin()
        y_index = np.abs(y_coordinates).argmin()

        cut_x = np.zeros((x_coordinates.shape[0], x_coordinates.shape[0]), dtype=np.complex128)
        cut_y = np.zeros((y_coordinates.shape[0], y_coordinates.shape[0]), dtype=np.complex128)

        for i_mode in range(self.numberVectors()):
            eigenvalue = eigenvalues[i_mode]
            mode = self.vector(i_mode)
            cut_x[:, :] += eigenvalue * np.outer(mode[:, y_index].conj(), mode[:, y_index])
            cut_y[:, :] += eigenvalue * np.outer(mode[x_index, :].conj(), mode[x_index, :])

        return cut_x, cut_y