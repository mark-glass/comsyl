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


class DegreeOfCoherence(object):
    def __init__(self, mutual_intesity):
        self.mutual_intesity = mutual_intesity
        self._symmetric_displacement = None
        self._plane_for_hole_origin = None
        self._v_displacement = None
        self._smaller_displacement = None
        self._srw_vertical = None
        self._srw_horizontal = None

        self._normalize = False


        if mutual_intesity is not None:
            self._x_coordinates = self.mutual_intesity.xCoordinates()
            self._y_coordinates = self.mutual_intesity.yCoordinates()
        else:
            self._x_coordinates = None
            self._y_coordinates = None


    def evaluate(self, r1, r2):
        result = self.mutual_intesity.cachedEvaluate(r1, r2)

        if self._normalize:
            result /= np.sqrt(self.mutual_intesity.evaluateIntensity(r1) * self.mutual_intesity.evaluateIntensity(r2))

        return  result


    def evaluateDirect(self, r1, r2):
        result = self.mutual_intesity.evaluate(r1, r2)

        if self._normalize:
            result /= np.sqrt(self.mutual_intesity.evaluateIntensity(r1) * self.mutual_intesity.evaluateIntensity(r2))

        return result

    def evaluateSRWHorizontal(self):

        x_coordinates = self.mutual_intesity.xCoordinates()
        y_coordinates = self.mutual_intesity.yCoordinates()
        x_index = x_coordinates.size/2
        y_index = y_coordinates.size/2

        res = np.zeros((len(x_coordinates), len(x_coordinates)), dtype=np.complex128)

        eigenvalues = self.mutual_intesity.eigenvalues()

        for i_e in range(self.mutual_intesity.numberVectors()):
            vector = self.mutual_intesity.vector(i_e)[:, y_index]
            res[:, :] += eigenvalues[i_e] * np.outer(vector.conj(), vector)

        return res

    def evaluateSRWVertical(self):

        x_coordinates = self.mutual_intesity.xCoordinates()
        y_coordinates = self.mutual_intesity.yCoordinates()
        x_index = x_coordinates.size/2
        y_index = y_coordinates.size/2

        res = np.zeros((len(y_coordinates), len(y_coordinates)), dtype=np.complex128)

        eigenvalues = self.mutual_intesity.eigenvalues()

        for i_e in range(self.mutual_intesity.numberVectors()):
            vector = self.mutual_intesity.vector(i_e)[x_index, :]
            res[:, :] += eigenvalues[i_e] * np.outer(vector.conj(), vector)

        return res

    def _normalizeMutualCoherenceFunction(self, in_array, sign_x, sign_y):

        if self._normalize:
            for i_x, x in enumerate(self._x_coordinates):
                for i_y, y in enumerate(self._y_coordinates):
                    r1 = np.array([x, y])
                    r2 = np.array([sign_x * x, sign_y * y])
                    in_array[i_x, i_y] /= np.sqrt(self.mutual_intesity.evaluateIntensity(r1) *
                                                  self.mutual_intesity.evaluateIntensity(r2))

        return in_array

    def planeForOriginHole(self):
        if self._plane_for_hole_origin is None:
            self._plane_for_hole_origin = self.planeForFixedR1(self._x_coordinates, self._y_coordinates, np.array([0.0, 0.0]))

        return  self._plane_for_hole_origin

    def planeForFixedR1(self, x_coordinates, y_coordinats, r_1):
        values = np.zeros((x_coordinates.shape[0], y_coordinats.shape[0]), dtype=np.complex128)
        for i_x, x in enumerate(x_coordinates):
            for i_y, y in enumerate(y_coordinats):
                r_2 = np.array([x, y])
                values[i_x, i_y] = self.evaluate(r_1, r_2)

        return values

    def symmetricDisplacement(self):
        if self._symmetric_displacement is None:
            self._symmetric_displacement = self.mutual_intesity.evaluateInversionCut()
            self._symmetric_displacement = self._normalizeMutualCoherenceFunction(self._symmetric_displacement, sign_x=-1.0, sign_y=-1.0)

        return self._symmetric_displacement

    def vDisplacement(self):
        if self._v_displacement is None:
            self._v_displacement = self.mutual_intesity.evaluateVCut()
            self._v_displacement = self._normalizeMutualCoherenceFunction(self._v_displacement, sign_x=-1.0, sign_y=1.0)

        return self._v_displacement

    def smallerDisplacement(self):
        if self._smaller_displacement is None:
            self._smaller_displacement = self.mutual_intesity.evaluateSmallerCut()
            self._smaller_displacement = self._normalizeMutualCoherenceFunction(self._smaller_displacement, sign_x=1.0, sign_y=-1.0)

        return self._smaller_displacement

    def SRWHorizontal(self):
        if self._srw_horizontal is None:
            self._srw_horizontal = self.evaluateSRWHorizontal()

        return self._srw_horizontal

    def SRWVertical(self):
        if self._srw_vertical is None:
            self._srw_vertical = self.evaluateSRWVertical()

        return self._srw_vertical

    def plotPlaneForOriginHole(self):
        from comsyl.math.utils import plotSurface
        plotSurface(self._x_coordinates, self._y_coordinates, np.abs(self.planeForOriginHole()))

    def plotSymmetricDisplacement(self):
        from comsyl.math.utils import plotSurface
        plotSurface(self._x_coordinates, self._y_coordinates, np.abs(self.symmetricDisplacement()))

    def plotVDisplacement(self):
        from comsyl.math.utils import plotSurface
        plotSurface(self._x_coordinates, self._y_coordinates, np.abs(self.vDisplacement()))

    def plotSmallerDisplacement(self):
        from comsyl.math.utils import plotSurface
        plotSurface(self._x_coordinates, self._y_coordinates, np.abs(self.smallerDisplacement()))

    def plotSRWHorizontal(self):
        from comsyl.math.utils import plotSurface
        plotSurface(self._x_coordinates, self._x_coordinates, np.abs(self.SRWHorizontal()))

    def plotSRWVertical(self):
        from comsyl.math.utils import plotSurface
        plotSurface(self._y_coordinates, self._y_coordinates, np.abs(self.SRWVertical()))


    def save(self, filename):

        np.savez(filename,
                 x_coordinates=self._x_coordinates,
                 y_coordinates=self._y_coordinates,
                 plane_for_hole_origin=self.planeForOriginHole(),
                 symmetric_displacement=self.symmetricDisplacement(),
                 v_displacement=self.vDisplacement(),
                 smaller_displacement=self.smallerDisplacement())

        print("Saved complex degree to %s" % filename)

    def load(self, filename):
        self.mutual_intesity = None

        file_content = np.load(filename)

        self._x_coordinates = file_content["x_coordinates"]
        self._y_coordinates = file_content["y_coordinates"]
        self._plane_for_hole_origin = file_content["plane_for_hole_origin"]
        self._symmetric_displacement = file_content["symmetric_displacement"]
        self._v_displacement = file_content["v_displacement"]
        self._smaller_displacement = file_content["smaller_displacement"]