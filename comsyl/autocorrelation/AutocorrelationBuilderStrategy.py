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
from comsyl.mathcomsyl.Convolution import Convolution
from comsyl.utils.Logger import log

class AutocorrelationBuilderStrategy(object):
    def __init__(self, x_coordinates, y_coordinates, density, field_x_coordinates, field_y_coordinates, weighted_fields):
        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates
        self._density = density

        self._weigthed_fields = weighted_fields
        self._setActiveField(0)

        self._field_tmp = np.zeros_like(self._field)
        self._field_tmp_x = np.zeros(self._field.shape[0], dtype=np.complex128)
        self._field_tmp_y = np.zeros(self._field.shape[1], dtype=np.complex128)
        self._field_x_coordinates = field_x_coordinates
        self._field_y_coordinates = field_y_coordinates

        self._last_r_1_x = None
        self._last_r_1_y = None

        self._x_coordinates_i_min = -(self._x_coordinates.shape[0] + 1)/2
        self._x_coordinates_i_max = (self._x_coordinates.shape[0] + 1)/2
        self._x_coordinates_step_width = self._x_coordinates[1] - self._x_coordinates[0]

        self._y_coordinates_i_min = -(self._y_coordinates.shape[0] + 1)/2
        self._y_coordinates_i_max = (self._y_coordinates.shape[0] + 1)/2
        self._y_coordinates_step_width = self._y_coordinates[1] - self._y_coordinates[0]

        self._grid_area = self._x_coordinates_step_width * self._y_coordinates_step_width

        log("Calculating density (integration part)")
        self._setRhoR(self._field_x_coordinates, self._field_y_coordinates)
        self._rho_phase_tmp = np.zeros_like(self._rho)

        self._convolution = Convolution()

        self._postInit()

    def _setActiveField(self, index_field):
        field = self._weigthed_fields[index_field, :, :]

        self._field = field
        self._field_conj = self._field.conj()
        self._field_reverse = self._field[::-1, ::-1]
        self._field_reverse_conj = self._field_reverse.conj()

    def numberFields(self):
        return self._weigthed_fields.shape[0]

    def _postInit(self):
        return

    def evaluate(self, r_1, r_2):
        raise NotImplementedError("Must implement.")

    def evaluateAllR_2(self, r_1):
        raise NotImplementedError("Must implement.")

    def _setRhoR(self, r_x, r_y):
        self._rho = np.zeros((r_x.shape[0], r_y.shape[0]), dtype=np.complex128)

        for i_x, x in enumerate(r_x):
            for i_y, y in enumerate(r_y):
                res = self._density.integrationPartGaussian(0.0, 0.0,
                                                            x=x,
                                                            y=y)
                self._rho[i_x, i_y] = res

    def _setRhoPhase(self, r_x, r_y):
        self._rho_phase_x = np.zeros((r_x.shape[0]), dtype=np.complex128)
        self._rho_phase_y = np.zeros((r_y.shape[0]), dtype=np.complex128)

        for x in r_x:
            i_x = self.xIndexByCoordinate(x)
            dr = np.array([x, 0.0])
            #TODO: why tranpose?
            self._rho_phase_x[i_x] = self._density.expPhaseMatrix().transpose().dot(dr)[0]
#            print("x", self._rho_phase_x[i_x])

        for y in r_y:
            i_y = self.yIndexByCoordinate(y)
            dr = np.array([0.0, y])
            #TODO: why tranpose?
            self._rho_phase_y[i_y] = self._density.expPhaseMatrix().transpose().dot(dr)[1]
#            print("y", self._rho_phase_y[i_y])

    def relativeShiftedCopy(self, i_x_shift, i_y_shift, field, shifted_field):

        shifted_field[:, :] = 0.0

        dim_x = field.shape[0]
        dim_y = field.shape[1]

        if i_x_shift > 0:
            x_start = 0
            x_end = dim_x - i_x_shift -1
            x_shift_start = i_x_shift
       #     print("x",x_start,x_end,i_x_shift, x_shift_start)
        else:
            x_start = -i_x_shift
            x_end = dim_x - 1
            x_shift_start = 0

        if i_y_shift > 0:
            y_start = 0
            y_end = dim_y - i_y_shift - 1
            y_shift_start = i_y_shift
       #     print("y",y_start,y_end, i_y_shift, y_shift_start)
        else:
            y_start = -i_y_shift
            y_end = dim_y - 1
            y_shift_start = 0

        x_shift_end = x_end + x_shift_start - x_start
        y_shift_end = y_end + y_shift_start - y_start

        shifted_field[x_shift_start:x_shift_end+1, y_shift_start:y_shift_end+1] = field[x_start:x_end+1, y_start:y_end+1]
      #  print(i_x_shift, i_y_shift,x_shift_start,x_shift_end, y_shift_start,y_shift_end,x_start,x_end, y_start,y_end, field.shape)

    def indexByCoordinate(self, coordinates, x):
        return np.abs(coordinates+x).argmin()

    def xIndexByCoordinate(self, x):
        return np.abs(self._x_coordinates-x).argmin() - int(self._x_coordinates.shape[0]/2)

    def yIndexByCoordinate(self, y):
        return np.abs(self._y_coordinates-y).argmin() - int(self._y_coordinates.shape[0]/2)

    def xIndexByCoordinateFast(self, x):
        return np.abs(self._x_coordinates-x).argmin()

    def yIndexByCoordinateFast(self, y):
        return np.abs(self._y_coordinates-y).argmin()

    def xFieldIndexByCoordinate(self, x):
        return np.abs(self._field_x_coordinates-x).argmin()

    def yFieldIndexByCoordinate(self, y):
        return np.abs(self._field_y_coordinates-y).argmin()

    def setAllCoordinates(self, x_coordinates,y_coordinates):

        self._density.setAllStaticPartCoordinates(x_coordinates, y_coordinates)

        self.i_all_coordiantes_x = np.array([self.xIndexByCoordinateFast(-x) for x in x_coordinates]) + self._x_coordinates_i_min
        self.i_all_coordiantes_y = np.array([self.yIndexByCoordinateFast(-y) for y in y_coordinates]) + self._y_coordinates_i_min

        self.i_all_coordiantes_x=np.array(self.i_all_coordiantes_x, dtype=np.int)
        self.i_all_coordiantes_y=np.array(self.i_all_coordiantes_y, dtype=np.int)

    def calculateIntensity(self):

        convoluted_intensity = None

        for i_field in range(self.numberFields()):
            self._setActiveField(i_field)
            field_intensity = np.abs(self._field)**2

            # TODO: use direct convolve when sampling rate is low.
            tmp = self._convolution.convolve2D(field_intensity, self._rho)
            if convoluted_intensity is None:
                convoluted_intensity = tmp
            else:
                convoluted_intensity += tmp

        convoluted_intensity *= self._grid_area
        convoluted_intensity *= self._density._static_part_prefactor

        return convoluted_intensity

    def setDoNotUseConvolutions(self, do_not_use_convolutions):
        self._do_not_use_convolutions = do_not_use_convolutions

        if self._do_not_use_convolutions:
            log("Never use convolutions")
        else:
            log("Using convolutions if possible")

    def doNotUseConvolutions(self):
        return self._do_not_use_convolutions
