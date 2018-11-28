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

from comsyl.mathcomsyl.utils import getFWHM, createGaussian2D
from comsyl.utils.Logger import log


class GaussianWavefrontBuilder():

    def fromWavefront(self, wavefront, info):
        e_field = wavefront.E_field_as_numpy()[0, :, :, 0]

        x_coordinates = wavefront.absolute_x_coordinates()
        y_coordinates = wavefront.absolute_y_coordinates()

        x_index = np.abs(x_coordinates).argmin()
        y_index = np.abs(y_coordinates).argmin()

        x_fwhm = getFWHM(x_coordinates, np.abs(e_field[:, y_index])**1)
        y_fwhm = getFWHM(y_coordinates, np.abs(e_field[x_index, :])**1)

        sigma_x = x_fwhm / (2 * np.sqrt(2 * np.log(2) ))
        sigma_y = y_fwhm / (2 * np.sqrt(2 * np.log(2) ))

        #sigma_x /= np.sqrt(2.0)
        #sigma_y /= np.sqrt(2.0)

        log(">>>")
        log(">>>Creating Gaussian wavefront using sigma_x: %e and sigma_y: %e" % (sigma_x, sigma_y))
        log(">>>")

        info.set("gaussian_wavefront_sigma_x", str(sigma_x))
        info.set("gaussian_wavefront_sigma_y", str(sigma_y))

        new_e_field = np.zeros_like(e_field)
        new_e_field[:, :] = createGaussian2D(sigma_x, sigma_y, x_coordinates, y_coordinates)

        new_wavefront = wavefront.toNumpyWavefront().clone()
        new_wavefront._e_field[0, :, :, 0] = new_e_field[:, :]

        return new_wavefront
