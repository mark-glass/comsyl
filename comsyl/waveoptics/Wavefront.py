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
from comsyl.mathcomsyl.utils import minAndMaxRow

class Wavefront(object):
    def minimal_x_coodinate(self):
        return min(self.absolute_x_coordinates())

    def maximal_x_coodinate(self):
        return max(self.absolute_x_coordinates())

    def minimal_y_coodinate(self):
        return min(self.absolute_y_coordinates())

    def maximal_y_coodinate(self):
        return max(self.absolute_y_coordinates())

    def x_stepwidth(self):
        return (self.x_end() - self.x_start()) / float(self.dim_x() - 1)

    def x_index_to_x(self, x_index):
        return self.x_start() + x_index * self.x_stepwidth()

    def y_index_to_y(self, y_index):
        return self.y_start() + y_index * self.y_stepwidth()

    def y_stepwidth(self):
        return (self.y_end() - self.y_start()) / float(self.dim_y() - 1)

    def absolute(self, x,y):
        absolute_x = self.x_start() + x * self.x_stepwidth()
        absolute_y = self.y_start() + y * self.y_stepwidth()

        return [absolute_x,absolute_y]

    def absolute_x_coordinates(self):
        x_coordinates = np.array([self.x_start() + x * self.x_stepwidth() for x in range(self.dim_x())])
        return x_coordinates

    def absolute_y_coordinates(self):
        y_coordinates = np.array([self.y_start() + y * self.y_stepwidth() for y in range(self.dim_y())])
        return y_coordinates

    def absolute_theta_angles(self):
        """
        Angles z with x.
        :return:
        """
        theta_angles = np.array([x / self.z() for x in self.absolute_x_coordinates()])
        theta_angles = theta_angles[theta_angles>=0.0]
        return theta_angles

    def absolute_phi_angles(self):
        """
        Angles z with y.
        :return:
        """
        phi_angles = np.linspace(0.0,2.0 * np.pi, len(self.absolute_x_coordinates()))
        return phi_angles

    def includesCoordinate(self,x,y):
        if self.x_start() <= x <= self.x_end() and self.y_start() <= y <= self.y_end():
            return True

        return False

    def interpolate(self, index_energy):
        import scipy.interpolate

        x = self.absolute_x_coordinates()
        y = self.absolute_y_coordinates()

        e_field = self.E_field_as_numpy()

        s_re_x = scipy.interpolate.RectBivariateSpline(x,y, e_field[index_energy,:,:,0].real)
        s_im_x = scipy.interpolate.RectBivariateSpline(x,y, e_field[index_energy,:,:,0].imag)

        s_re_y = 1#scipy.interpolate.interp2d(x,y, e_field[index_energy,:,:,1].real)
        s_im_y = 1#scipy.interpolate.interp2d(x,y, e_field[index_energy,:,:,1].imag)


        s = lambda x, y: (s_re_x(x,y) + 1j*s_im_x(x,y), None)#s_re_y(x,y) + 1j*s_im_y(x,y))

        return s

    def interpolateAngular(self, index_energy):
        f_cartesian = self.interpolate(index_energy)

        f_angular = lambda theta,phi: f_cartesian(self.z() * np.tan(theta) * np.cos(phi),
                                                  self.z() * np.tan(theta) * np.sin(phi))

        return f_angular

    def intensity_as_numpy(self):
        return np.abs(self.E_field_as_numpy())**2

    def intensity(self, x, y):
        e_field = self.efield(x,y,0)
        return abs(e_field[0])**2+abs(e_field[1])**2

    def intensity_at_x(self, x):
        intensity = []
        for y in range(self.dim_y()):
            intensity.append(self.intensity(x,y))

        return intensity

    def intensity_at_y(self, y):
        intensity = []
        for x in range(self.dim_x()):
            intensity.append(self.intensity(x,y))

        return intensity

    def intensity_plane(self):
        E_field = self.E_field_as_numpy()
        intensity = abs(E_field) ** 2
        intensity = intensity.sum(3)

        return intensity

    def interpolatedIntensity(self, x_coordinates, y_coordinates):

        dim_x = len(x_coordinates)
        dim_y = len(y_coordinates)

        intensity = np.zeros((self.numberEnergies(), 2, dim_x, dim_y), dtype=np.complex128)

        for index_energy in range(self.numberEnergies()):
            interpolante = self.interpolate(index_energy)
            value = interpolante(x_coordinates, y_coordinates)
            intensity[index_energy, :, :, :] = np.abs(value)**2

        return intensity.real

    def interpolatedPhase(self, x_coordinates, y_coordinates):

        dim_x = len(x_coordinates)
        dim_y = len(y_coordinates)

        phase = np.zeros((self.numberEnergies(), 2, dim_x, dim_y), dtype=np.complex128)

        for index_energy in range(self.numberEnergies()):
            interpolante = self.interpolate(index_energy)
            value = interpolante(x_coordinates, y_coordinates)  
            phase[index_energy, :, :, :] = np.angle(value)# / np.abs(value)

        return phase

    def determineCoveredCoordinates(self, coordinates, new_coordinates):
        covered_coordinates = new_coordinates[(new_coordinates>coordinates.min()) &
                                              (new_coordinates<coordinates.max())]
        return covered_coordinates

    def coveredInterpolation(self, new_x_coordinates, new_y_coordinates):

        covered_x_coordinates = self.determineCoveredCoordinates(self.absolute_x_coordinates(),
                                                                 new_x_coordinates)

        covered_y_coordinates = self.determineCoveredCoordinates(self.absolute_y_coordinates(),
                                                                 new_y_coordinates)


        f_wfr = self.interpolate(0)

        #TODO: polarization
        interior = f_wfr(covered_x_coordinates, covered_y_coordinates)[0]

        covered_mode = np.zeros((new_x_coordinates.shape[0], new_y_coordinates.shape[0]), dtype=np.complex128)

        x_min = self.absolute_x_coordinates().min()
        y_min = self.absolute_y_coordinates().min()

        i_x_min = np.abs(new_x_coordinates-x_min).argmin()
        i_y_min = np.abs(new_y_coordinates-y_min).argmin()

        i_x_max = i_x_min + interior.shape[0]
        i_y_max = i_y_min + interior.shape[1]

        covered_mode[i_x_min:i_x_max, i_y_min:i_y_max] = interior[:, :]

        return covered_mode

    def add(self, add_wavefront):
        print("Adding wavefront")

        #print("self")
        #self.printInfo()
        #print("summand")
        #add_wavefront.printInfo()

        x_start = min(self.x_start(), add_wavefront.x_start())
        x_end = max(self.x_end(), add_wavefront.x_end())
        y_start = min(self.y_start(), add_wavefront.y_start())
        y_end = max(self.y_end(), add_wavefront.y_end())

        x_stepwidth = max(self.x_stepwidth(), add_wavefront.x_stepwidth())
        y_stepwidth = max(self.y_stepwidth(), add_wavefront.y_stepwidth())

        dim_x = int((x_end - x_start) / x_stepwidth)+1
        dim_y = int((y_end - y_start) / y_stepwidth)+1

        e_field = np.zeros((self.numberEnergies(), dim_x, dim_y, 2), dtype=np.complex128)


        wavefront_sum = NumpyWavefront(e_field, x_start, x_end, y_start, y_end, self.z(), self.energies())

        #print("Add info")
        #print("x_start", x_start)
        #print("x_end", x_end)
        #print("y_start", y_start)
        #print("y_end", y_end)
        #print("x_stepwidth", x_stepwidth)
        #print("y_stepwidth", y_stepwidth)
        #print("dim_x", dim_x)
        #print("dim_y", dim_y)

        print("Addition I_X ", end="")
        for index_energy in range(self.numberEnergies()):
            summand_wavefronts = [self, add_wavefront]
            for summand in summand_wavefronts:
                interpolante = summand.interpolate(index_energy)
                value = interpolante(wavefront_sum.absolute_x_coordinates(), wavefront_sum.absolute_y_coordinates())
                for i_x, x_cooridinate in enumerate(wavefront_sum.absolute_x_coordinates()):
                    if i_x%100 ==0:
                        print(" ",i_x , end="")
                        sys.stdout.flush()
                    for i_y, y_cooridinate in enumerate(wavefront_sum.absolute_y_coordinates()):
                        if summand.includesCoordinate(x_cooridinate, y_cooridinate):
                            e_field[index_energy,i_x, i_y, 0] = e_field[index_energy,i_x, i_y, 0] \
                                                                + value[0][i_x][i_y]
                            # e_field[index_energy,i_x, i_y, 1] = e_field[index_energy,i_x, i_y, 1] \
                            #                                     + value[1][i_x][i_y]

        #wavefront_sum.printInfo()

        return wavefront_sum

    def mul(self, add_wavefront):
        print("Multiply wavefront")

        x_start = min(self.x_start(), add_wavefront.x_start())
        x_end = max(self.x_end(), add_wavefront.x_end())
        y_start = min(self.y_start(), add_wavefront.y_start())
        y_end = max(self.y_end(), add_wavefront.y_end())

        x_stepwidth = max(self.x_stepwidth(), add_wavefront.x_stepwidth())
        y_stepwidth = max(self.y_stepwidth(), add_wavefront.y_stepwidth())

        dim_x = int((x_end - x_start) / x_stepwidth)+1
        dim_y = int((y_end - y_start) / y_stepwidth)+1

        e_field = np.ones((self.numberEnergies(), dim_x, dim_y, 2), dtype=np.complex128)


        wavefront_sum = NumpyWavefront(e_field, x_start, x_end, y_start, y_end, self.z())

        for index_energy in range(self.numberEnergies()):
            summand_wavefronts = [self, add_wavefront]
            for summand in summand_wavefronts:
                interpolante = summand.interpolate(index_energy)
                value = interpolante(wavefront_sum.absolute_x_coordinates(), wavefront_sum.absolute_y_coordinates())
                for i_x, x_cooridinate in enumerate(wavefront_sum.absolute_x_coordinates()):
                    for i_y, y_cooridinate in enumerate(wavefront_sum.absolute_y_coordinates()):
                        if summand.includesCoordinate(x_cooridinate, y_cooridinate):
                            e_field[index_energy,i_x, i_y, 0] = e_field[index_energy,i_x, i_y, 0] \
                                                                * value[0][i_x][i_y]
                            e_field[index_energy,i_x, i_y, 1] = e_field[index_energy,i_x, i_y, 1] \
                                                                * value[1][i_x][i_y]

        return wavefront_sum

    def convolve(self, f):
        from scipy.signal import convolve

        e_field = self.E_field_as_numpy()
        f_data = np.zeros((self.dim_x(), self.dim_y()))
        result = np.zeros((self.numberEnergies(), self.dim_x(), self.dim_y(), 2), dtype=np.complex128)

        print("Convolve I_X ", end="")
        for i_x, x_cooridinate in enumerate(self.absolute_x_coordinates()):
            if i_x%100 ==0:
                print(" ",i_x , end="")
            for i_y, y_cooridinate in enumerate(self.absolute_y_coordinates()):
                f_data[i_x, i_y] = f(x_cooridinate,y_cooridinate)

        for index_energy in range(self.numberEnergies()):
            for pol in (0,1):
                print("Convolving pol", pol)
                #r = convolve(f_data, f_data,mode='same')
                r = convolve(e_field[index_energy,:,:,pol].real, f_data,mode='same')
                r = r + 1j *convolve(e_field[index_energy,:,:,pol].imag, f_data,mode='same')

                for i_x, x_cooridinate in enumerate(self.absolute_x_coordinates()):
                    for i_y, y_cooridinate in enumerate(self.absolute_y_coordinates()):
                        result[index_energy, i_x , i_y , pol] = r[i_x,i_y]

        convolved_wavefront = NumpyWavefront(result, self.x_start(), self.x_end(), self.y_start(), self.y_end(), self.z())

        return convolved_wavefront

    def toNumpyWavefront(self):
        return NumpyWavefront(self.E_field_as_numpy().copy(),
                              self.x_start(), self.x_end(), self.y_start(), self.y_end(), self.z(),
                              self.energies())

    def printInfo(self):
        print("Wavefront info start")
        print("--x_start:",self.x_start())
        print("--x_end:",self.x_end())
        print("--y_start:",self.y_start())
        print("--y_end:",self.y_end())
        print("--x_stepwidth", self.x_stepwidth())
        print("--y_stepwidth", self.y_stepwidth())
        print("--dim_x", self.dim_x())
        print("--dim_y", self.dim_y())
        print("Wavefront info end")

    def showEField(self):
        from comsyl.mathcomsyl.utils import plotSurface
        x = np.array(self.absolute_x_coordinates())
        y = np.array(self.absolute_y_coordinates())
        z = self.E_field_as_numpy()[0, :, :, 0]
        intensity = np.abs(z)
        plotSurface(x, y, intensity)
        plotSurface(x, y, intensity, False)

        phase = np.angle(z)
        phase[intensity < intensity.max()/10.0] = 0.0

        plotSurface(x,y,phase)
        #plotSurface(x,y,phase, False)

    def wavelengths(self):
        energies = self.energies()
        wavelength = np.zeros_like(energies)

        for i_e, energy in enumerate(energies):
            wavelength[i_e] = 1239.84193e-9 / float(energy)

        return wavelength

    def wavenumbers(self):
        wavenumbers = np.zeros_like(self.wavelengths())

        for i_e, wavelength in enumerate(self.wavelengths()):
            wavenumbers[i_e] = (2 * np.pi) / wavelength

        return wavenumbers

class NumpyWavefront(Wavefront):
    def __init__(self, e_field, x_start, x_end, y_start, y_end, z, energies):
        """

        :param e_field: (Energy, x, y, polarization)
        :param x_start:
        :param x_end:
        :param y_start:
        :param y_end:
        :param z:
        :return:
        """
        self._e_field = e_field
        self._x_start = x_start
        self._x_end = x_end
        self._y_start = y_start
        self._y_end = y_end
        self._z = z
        self._energies = np.array(energies).copy()

    def numberEnergies(self):
        return self.dim_energy()

    def energies(self):
        return self._energies

    def dim_x(self):
        return self._e_field.shape[1]

    def x_start(self):
        return self._x_start

    def x_end(self):
        return self._x_end

    def dim_y(self):
        return self._e_field.shape[2]

    def y_start(self):
        return self._y_start

    def y_end(self):
        return self._y_end

    def z(self):
        return self._z

    def dim_energy(self):
        return self._e_field.shape[0]

    def efield(self, x, y, index_energy):
        e_horizontal = self._e_field[index_energy,x,y,0]
        e_vertical = self._e_field[index_energy,x,y,1]

        return [e_horizontal, e_vertical]

    def conjugate(self):
        self._e_field = self._e_field.conj()

    def E_field_as_numpy(self):
        return self._e_field

    def extend(self, new_x_start, new_x_end, new_y_start, new_y_end):
        print("Extending wavefront")
        new_dim_x = int((new_x_end-new_x_start) / self.x_stepwidth())
        new_dim_y = int((new_y_end-new_y_start) / self.y_stepwidth())

        real_new_x_end = new_x_start + new_dim_x * self.x_stepwidth()
        real_new_x_end = new_y_start + new_dim_y * self.y_stepwidth()

        new_e_field = np.zeros((self.numberEnergies(), new_dim_x, new_dim_y, 2),dtype=np.complex128)

        # TODO: upto +- 1
        i_x_start = int((self._x_start - new_x_start) / self.x_stepwidth())
        i_x_end = int((self._x_end - new_x_start) / self.x_stepwidth())
        i_y_start = int((self._y_start - new_y_start) / self.y_stepwidth())
        i_y_end = int((self._y_end - new_y_start) / self.y_stepwidth())

        for index_energy in range(self.numberEnergies()):
            for i_x in range(i_x_start, i_x_end):
                for i_y in range(i_y_start, i_y_end):
                    new_e_field[index_energy, i_x, i_y, 0] = self._e_field[index_energy, i_x-i_x_start, i_y-i_y_start, 0]
                    new_e_field[index_energy, i_x, i_y, 1] = self._e_field[index_energy, i_x-i_x_start, i_y-i_y_start, 1]

        self._e_field = new_e_field
        self._x_start = new_x_start
        self._x_end = new_x_end
        self._y_start = new_y_start
        self._y_end = new_y_end

    def shiftOrigin(self, x_shift, y_shift):
        self._x_start = self._x_start + self.x_stepwidth() *  x_shift
        self._x_end = self._x_end + self.x_stepwidth() * x_shift
        self._y_start = self._y_start + self.y_stepwidth() * y_shift
        self._y_end = self._y_end + self.y_stepwidth() * y_shift

    def asCenteredGrid(self, fixed_x_coordinates=None, fixed_y_coordinates=None, resample_x=1.0, resample_y=1.0):

        if fixed_x_coordinates is None:
            x_step = self.x_stepwidth()
            x_min = self.minimal_x_coodinate() + x_step
            x_max = self.maximal_x_coodinate() - x_step

            if self.dim_x()%2 == 0:
               dim_x = self.dim_x() - 1
            else:
               dim_x = self.dim_x()
            
            new_x = np.linspace(x_min, x_max, dim_x * resample_x)
        else:
            new_x = fixed_x_coordinates

        if fixed_y_coordinates is None:
            y_step = self.y_stepwidth()
            y_min = self.minimal_y_coodinate() + y_step
            y_max = self.maximal_y_coodinate() - y_step

            if self.dim_y()%2 == 0:
               dim_y = self.dim_y() - 1
            else:
               dim_y = self.dim_y()

            new_y = np.linspace(y_min, y_max, dim_y * resample_y)
        else:
            new_y = fixed_y_coordinates

        # TODO: only works for single energy grids
        if len(self.energies()) > 1:
            raise NotImplementedError("Can only center for single energy grids.")

        f_e_field = self.interpolate(index_energy=0)

        new_e_field = f_e_field(new_x, new_y)

        e_field = np.zeros((1, new_e_field[0].shape[0], new_e_field[0].shape[1], 2), dtype=new_e_field[0].dtype)

        e_field[0, :, :, 0] = new_e_field[0]
        e_field[0, :, :, 1] = new_e_field[1]

        centered_wavefront = NumpyWavefront(e_field=e_field,
                                            x_start=new_x.min(),
                                            x_end=new_x.max(),
                                            y_start=new_y.min(),
                                            y_end=new_y.max(),
                                            z=self.z(),
                                            energies=self.energies())

        return centered_wavefront

    def asEvenGridpointsGrid(self, resample_x=1.0, resample_y=1.0):
        if self.dim_x() % 2 == 1:
            new_dim_x = self.dim_x() - 1
        else:
            new_dim_x = self.dim_x()

        if self.dim_y() % 2 == 1:
            new_dim_y = self.dim_y() - 1
        else:
            new_dim_y = self.dim_y()

        if new_dim_x == self.dim_x() and new_dim_y == self.dim_y():
            return self

        new_wavefront = self.asFixedSize(new_dim_x=new_dim_x * resample_x,
                                         new_dim_y=new_dim_y * resample_y)

        return new_wavefront

    def findSupportIndices(self, twod_array, relative_tolerance=0.001):

        abs_array = np.abs(twod_array)

        tolerance = relative_tolerance * abs_array.max()

        for i_row_min in range(twod_array.shape[0]):
            if not np.all(abs_array[i_row_min,:] < tolerance):
                break

        for i_row_max in range(twod_array.shape[0]-1,-1,-1):
            print(i_row_max)
            if not np.all(abs_array[i_row_max,:] < tolerance):
                break

        for i_column_min in range(twod_array.shape[1]):
            if not np.all(abs_array[:,i_column_min] < tolerance):
                break

        for i_column_max in range(twod_array.shape[1]-1,-1,-1):
            if not np.all(abs_array[:,i_column_max] < tolerance):
                break

        return i_row_min, i_row_max, i_column_min, i_column_max

    def shrinkedToSupport(self):
        # TODO: polarization
        i_row_min, i_row_max, i_column_min, i_column_max = self.findSupportIndices(self._e_field[0,:,:,0])

        x_min = self.x_index_to_x(i_row_min)
        x_max = self.x_index_to_x(i_row_max)
        dim_x = i_row_max - i_row_min

        y_min = self.y_index_to_y(i_column_min)
        y_max = self.y_index_to_y(i_column_max)
        dim_y = i_column_max - i_column_min

        print(x_min, x_max, dim_x)
        print(y_min, y_max, dim_y)

        return self.onDomain(x_min, x_max, dim_x,
                             y_min, y_max, dim_y)

    def onDomain(self, x_min, x_max, dim_x, y_min, y_max, dim_y):
        new_x = np.linspace(x_min, x_max, dim_x)
        new_y = np.linspace(y_min, y_max, dim_y)

        # TODO: only works for single energy grids
        if len(self.energies()) > 1:
            raise NotImplementedError("Can only center for single energy grids.")

        new_e_field = self.coveredInterpolation(new_x, new_y)

        e_field = np.zeros((1, new_e_field.shape[0], new_e_field.shape[1], 2), dtype=new_e_field.dtype)

        e_field[0, :, :, 0] = new_e_field[:, :]
        # TODO: polarization
        e_field[0, :, :, 1] = 0

        new_wavefront = NumpyWavefront(e_field=e_field,
                                       x_start=x_min,
                                       x_end=x_max,
                                       y_start=y_min,
                                       y_end=y_max,
                                       z=self.z(),
                                       energies=self.energies())

        return new_wavefront

    def asFixedSize(self, new_dim_x, new_dim_y):
        x_min = self.minimal_x_coodinate()
        x_max = self.maximal_x_coodinate()

        y_min = self.minimal_y_coodinate()
        y_max = self.maximal_y_coodinate()

        return self.onDomain(x_min, x_max, new_dim_x,
                             y_min, y_max, new_dim_y)

    def zeroPadded(self, zero_padding_factor_x=2, zero_padding_factor_y=1.5):
        new_x_min = self.minimal_x_coodinate() * zero_padding_factor_x
        new_x_max = self.maximal_x_coodinate() * zero_padding_factor_x
        new_dim_x = self.dim_x() * 2

        new_y_min = self.minimal_y_coodinate() * zero_padding_factor_y
        new_y_max = self.maximal_y_coodinate() * zero_padding_factor_y
        new_dim_y = self.dim_y() * 2

        return self.onDomain(new_x_min, new_x_max, new_dim_x,
                             new_y_min, new_y_max, new_dim_y)

    def __eq__(self, other):
        equals = np.linalg.norm(self._e_field - other._e_field) < 1e-10
        equals = equals and self._x_start == other._x_start
        equals = equals and self._x_end == other._x_end
        equals = equals and self._y_start == other._y_start
        equals = equals and self._y_end == other._y_end
        equals = equals and self._z == other._z
        equals = equals and np.linalg.norm(self._energies - other._energies) < 1e-10

        return equals

    def __ne__(self, other):
        return not self.__eq__(other)

    def asNumpyArray(self):
        coordinates = np.array([self.x_start(), self.x_end(), self.y_start(), self.y_end(), self.z()])

        return self._e_field.copy(), coordinates.copy(), self.energies().copy()

    @staticmethod
    def fromNumpyArray(e_field, coordinates, energies):
        return NumpyWavefront(e_field, coordinates[0], coordinates[1], coordinates[2], coordinates[3], coordinates[4], energies)

    @staticmethod
    def fromNumpyWavefront(numpy_wavefront):
        data = numpy_wavefront.asNumpyArray()

        return NumpyWavefront(data[0], data[1][0], data[1][1], data[1][2], data[1][3], data[1][4], data[2])

    def clone(self):
        return NumpyWavefront.fromNumpyWavefront(self)

    def save(self, filename):
        as_numpy_array = self.asNumpyArray()
        np.savez(filename,
                 e_field=as_numpy_array[0],
                 coordinates=as_numpy_array[1],
                 energies=as_numpy_array[2])

    @staticmethod
    def load(filename):
        file_content = np.load(filename)

        e_field = file_content["e_field"]
        coordinates = file_content["coordinates"]
        energies = file_content["energies"]

        return NumpyWavefront.fromNumpyArray(e_field, coordinates, energies)


class SRWWavefront(Wavefront):
    def __init__(self, srw_wavefront):
        self._srw_wavefront = srw_wavefront

    def SRWWavefront(self):
        return self._srw_wavefront

    def numberEnergies(self):
        return self._srw_wavefront.mesh.ne

    def energies(self):
        energies = np.linspace(self._srw_wavefront.mesh.eStart,
                               self._srw_wavefront.mesh.eFin,
                               self.numberEnergies())
        return energies

    def dim_x(self):
        return self._srw_wavefront.mesh.nx

    def x_start(self):
        return self._srw_wavefront.mesh.xStart

    def x_end(self):
        return self._srw_wavefront.mesh.xFin

    def dim_y(self):
        return self._srw_wavefront.mesh.ny

    def y_start(self):
        return self._srw_wavefront.mesh.yStart

    def y_end(self):
        return self._srw_wavefront.mesh.yFin

    def z(self):
        return self._srw_wavefront.mesh.zStart

    def dim_energy(self):
        return self._srw_wavefront.mesh.ne

    def index(self, x, y, index_energy):
        index = 2*y*self.dim_x() * self.dim_energy() + 2*x * self.dim_energy() + 2*index_energy
        return index

    def efield(self, x, y, index_energy):
        index = self.index(x,y,index_energy)
        e_horizontal = self._srw_wavefront.arEx[index] + self._srw_wavefront.arEx[index+1] * 1j
        e_vertical = self._srw_wavefront.arEy[index] + self._srw_wavefront.arEy[index+1] * 1j

        return [e_horizontal, e_vertical]

    def _srw_array_to_numpy(self, srw_array):
        re=np.array(srw_array[::2], dtype=np.float)
        im=np.array(srw_array[1::2], dtype=np.float)

        e = re + 1j * im
        e=e.reshape((self.dim_y(),
                     self.dim_x(),
                     self.numberEnergies(),
                     1)
                    )

        e = e.swapaxes(0,2)

        return e

    def E_field_as_numpy(self):
        x_polarization = self._srw_array_to_numpy(self._srw_wavefront.arEx)
        y_polarization = self._srw_array_to_numpy(self._srw_wavefront.arEy)

        e_field = np.concatenate((x_polarization,y_polarization),3)

        return e_field

    def instantRadii(self):
        Rx = self._srw_wavefront.Rx
        dRx = self._srw_wavefront.dRx

        Ry = self._srw_wavefront.Ry
        dRy = self._srw_wavefront.dRy

        #srwl.SetRepresElecField(calc_wavefront._srw_wavefront, 'a')

        return Rx, dRx, Ry, dRy
