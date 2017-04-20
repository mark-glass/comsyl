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
import scipy.interpolate

def _determineCoveredCoordinates(coordinates, new_coordinates):
    covered_coordinates = new_coordinates[(new_coordinates>coordinates.min()) &
                                          (new_coordinates<coordinates.max())]
    return covered_coordinates

def coveredInterpolation(x, y, z, new_x_coordinates, new_y_coordinates, use_uncovered=False):
    # TODO: check vertical vs horizontal
    f_z = scipy.interpolate.RectBivariateSpline(x, y, z)

    if(new_x_coordinates.min() <= x.min() and new_x_coordinates.max() < x.max() and \
       new_y_coordinates.min() <= y.min() and new_y_coordinates.max() < y.max())\
            or use_uncovered == True:
        return f_z(new_x_coordinates, new_y_coordinates)

    covered_x_coordinates = _determineCoveredCoordinates(x, new_x_coordinates)
    covered_y_coordinates = _determineCoveredCoordinates(y, new_y_coordinates)

    interior = f_z(covered_x_coordinates, covered_y_coordinates)

    covered_interpolated = np.zeros((new_x_coordinates.shape[0], new_y_coordinates.shape[0]), dtype=z.dtype)

    x_min = x.min()
    y_min = y.min()

    i_x_min = np.abs(new_x_coordinates-x_min).argmin()
    i_y_min = np.abs(new_y_coordinates-y_min).argmin()

    i_x_max = i_x_min + interior.shape[0]
    i_y_max = i_y_min + interior.shape[1]

    covered_interpolated[i_x_min:i_x_max, i_y_min:i_y_max] = interior[:, :]

    #print(i_x_min,i_x_max, i_y_min,i_y_max)

    return covered_interpolated