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
import scipy.optimize as opt

try:
    import matplotlib.pyplot as plt
except:
    pass

def trapez2D(integrand, dx, dy):
    return np.trapz(np.trapz(integrand,dx=dx),dx=dy)

def norm1D(coordinates, values):
    return np.sqrt(np.trapz(np.abs(values)**2, coordinates))

def norm2D(x_coords, y_coords, i):
    s = np.zeros(i.shape[0])

    for k in range(i.shape[0]):
        s[k]=np.trapz(np.abs(i[k, :])**2, y_coords)

    res = np.trapz(s, x_coords)

    return np.sqrt(res)

def plot(x, y, limit=None, title=None):
    plt.plot(x, y)

    if limit is not None:
        plt.gca().set_xlim(-limit, limit)

    if title is not None:
        plt.title(title)

    plt.show()

def plotSurface(x, y, z, contour_plot=True, title=None,filename=None):
    from mpl_toolkits.mplot3d import Axes3D
    from scipy import meshgrid, array
    import scipy.interpolate

    fig = plt.figure()

    if contour_plot:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')

    x_coordinates = x
    y_coordinates = y

    f_int = scipy.interpolate.RectBivariateSpline(x_coordinates,y_coordinates, z)

    #x_coordinates = np.linspace(min(x_coordinates), max(x_coordinates), 400)
    #y_coordinates = np.linspace(min(y_coordinates), max(y_coordinates), 400)
    #z = f_int(x_coordinates, y_coordinates)

    if isinstance(z,np.ndarray):
        plane = []
        for i_x in range(len(x_coordinates)):
            for i_y in range(len(y_coordinates)):
                plane.append(float(z[i_x,i_y]))
    else:
        plane = z

    X, Y = meshgrid(x_coordinates,
                    y_coordinates)

    zs = array(plane)

    Z = z.transpose()# zs.reshape(X.shape)

    if contour_plot:
        CS = ax.contour(X, Y, Z)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
    else:
        ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X in plane')
    ax.set_ylabel('Y in plane')

    if not contour_plot:
        ax.set_zlabel('z')


    if title is not None:
        plt.title(title)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')

def plotTrajectory(trajectory):
    t = trajectory.t()

    plot(t, trajectory.x(), title="Horizontal position")
    plot(t, trajectory.y(), title="Vertical position")
    plot(t, trajectory.v_x(), title="Horizontal velocity")
    plot(t, trajectory.v_y(), title="Vertical velocity")

def plotMagneticfield(magnetic_field):
    z = magnetic_field.z()

    plot(z, magnetic_field.B_x(), title="Horizontal magnetic field")
    plot(z, magnetic_field.B_y(), title="Vertical magnetic field")


def createGaussian2D(sigma_x, sigma_y, x_coordinates, y_coordinates):
    f = np.zeros((x_coordinates.shape[0],
                  y_coordinates.shape[0]))

    for i_x, x in enumerate(x_coordinates):
        for i_y, y in enumerate(y_coordinates):
            f[i_x, i_y] = (1.0/(2.0*np.pi*sigma_x*sigma_y))*np.exp(-x**2/(2*sigma_x**2)-y**2/(2*sigma_y**2))

    return f

def createSinc2D(sigma_x, sigma_y, x_coordinates, y_coordinates, detune):
    f = np.zeros((x_coordinates.shape[0],
                  y_coordinates.shape[0]))

    for i_x, x in enumerate(x_coordinates):
        for i_y, y in enumerate(y_coordinates):
            f[i_x, i_y] = np.sinc(x**2 * sigma_x+y**2 * sigma_y + detune)

    return f

def axisCuts(x_coords, y_coords, af):
    i_mid_x = int(x_coords.shape[0]/2) - 1
    x_1 = x_coords[i_mid_x]
    x_2 = x_coords[i_mid_x+3]

    i_mid_y = int(y_coords.shape[0]/2) - 1
    y_1 = y_coords[i_mid_y]
    y_2 = y_coords[i_mid_y+3]

    f_x = np.zeros((x_coords.shape[0],) * 2,
                    dtype=np.complex128)

    f_y = np.zeros((x_coords.shape[0],) * 2,
                   dtype=np.complex128)

    for i_x_1 in range(x_coords.shape[0]):
        r_1 = np.array([x_coords[i_x_1], y_1])
        for i_x_2 in range(x_coords.shape[0]):
            r_2 = np.array([x_coords[i_x_2], y_2])

            f_x[i_x_1, i_x_2] = af.evaluate(r_1, r_2)

    for i_y_1 in range(y_coords.shape[0]):
        r_1 = np.array([x_1, y_coords[i_y_1]])
        for i_y_2 in range(y_coords.shape[0]):
            r_2 = np.array([x_2, y_coords[i_y_2]])

            f_y[i_y_1, i_y_2] = af.evaluate(r_1, r_2)

    plotSurface(x_coords, x_coords, f_x, False)
    plotSurface(y_coords, y_coords, f_y, False)

    f=np.zeros_like(f_x)
    g=np.zeros_like(f_x)

    for i_x_1 in range(x_coords.shape[0]):
        for i_y_1 in range(y_coords.shape[0]):
            r_1 = np.array([x_coords[i_x_1], y_coords[i_y_1]])
            r_2 = np.array([x_coords[i_x_1], y_coords[i_y_1]])

            f[i_x_1,i_y_1] = af.evaluate(r_1, r_2)
            g[i_x_1,i_y_1]=f_x[i_x_1, i_x_1] * f_y[i_y_1, i_y_1] / f[i_x_1,i_y_1]

    plotSurface(x_coords, x_coords, f, False)
    plotSurface(y_coords, y_coords, g, False)

def gauss(x, p): # p[0]==mean, p[1]==stdev
    return p[2]*(1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2)))

def _minOrMaxRow(min_or_max, data, norm_fraction):
    abs_data = np.abs(data)

    abs_sum = np.sum(abs_data)

    n_row = abs_data.shape[0]

    abs_sum_partial = 0.0

    if min_or_max == "min":
        row_sense = list(range(n_row))
    elif min_or_max == "max":
        row_sense = list(range(n_row))
        row_sense.reverse()
    else:
        raise NotImplementedError

    i_row_min_or_max = None
    for i_row in row_sense:
        abs_sum_partial += np.sum(abs_data[i_row,:])
        if(abs_sum_partial/abs_sum >= norm_fraction):
            i_row_min_or_max = i_row
            break

    return i_row_min_or_max

def minAndMaxRow(data, norm_fraction):
    i_row_min = _minOrMaxRow(min_or_max="min", data=data, norm_fraction=norm_fraction)
    i_row_max = _minOrMaxRow(min_or_max="max", data=data, norm_fraction=norm_fraction)

    return i_row_min, i_row_max

def getGaussianSigmaDirect(X, Y):
    xmin = X.min()
    xmax = X.max()
    N = X.shape[0]

    # Normalize to a proper PDF
    #Y /= ((xmax-xmin)/N)*Y.sum()

    # Fit a guassian
    p0 = [0, 1, 1] # Inital guess is a normal distribution
    errfunc = lambda p, x, y: gauss(x, p) - y # Distance to the target function
    p1, success = opt.leastsq(errfunc, p0[:], args=(X, Y))

    print(p1)
    fit_mu, fit_stdev, fit_a = p1
    return fit_stdev

def getFWHM(x, y):
    x_positive = x[x >= 0.0]
    x_negative = x[x < 0.0]

    y_positive = y[x >= 0.0]
    y_negative = y[x < 0.0]

    i_positive = np.abs((y_positive-y.max()/2.0)).argmin()
    i_negative = np.abs((y_negative-y.max()/2.0)).argmin()

    x_plus = x_positive[i_positive]
    x_minus = x_negative[i_negative]

    return x_plus - x_minus

def getFWHMLowerUpper(x, y):
    x_positive = x[x >= 0.0]
    x_negative = x[x < 0.0]

    y_positive = y[x >= 0.0]
    y_negative = y[x < 0.0]

    i_positive = np.abs((y_positive-y.max()/2.0)).argmin()
    i_negative = np.abs((y_negative-y.max()/2.0)).argmin()

    i_positive_lower = i_positive if y_positive[i_positive]-y.max()/2.0 < 0.0 else i_positive - 1
    i_negative_lower = i_negative if y_negative[i_negative]-y.max()/2.0 < 0.0 else i_negative + 1

    i_positive_upper = i_positive if y_positive[i_positive]-y.max()/2.0 > 0.0 else i_positive + 1
    i_negative_upper = i_negative if y_negative[i_negative]-y.max()/2.0 > 0.0 else i_negative - 1

    x_plus_lower = x_positive[i_positive_lower]
    x_minus_lower = x_negative[i_negative_lower]

    x_plus_upper = x_positive[i_positive_upper]
    x_minus_upper = x_negative[i_negative_upper]

    print(x_plus_lower, x_minus_lower, x_plus_upper, x_minus_upper)

    fwhm_lower = x_plus_lower - x_minus_lower
    fwhm_upper = x_plus_upper - x_minus_upper

    d_fwhm = (fwhm_upper-fwhm_lower)/2.0

    return fwhm_lower, fwhm_upper, fwhm_lower+d_fwhm, d_fwhm

def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def getGaussianSigma(x, y):
    fwhm_to_sigma = (2 * np.sqrt(2 * np.log(2) ))

    sigma_rough = getFWHM(x, y) / fwhm_to_sigma

    popt, pcov = opt.curve_fit(gaussian, x, y, p0=[1, 0, sigma_rough])
    return popt[2]

def sorted2DIndices(values_x, values_y):
    i_c = 0
    array_size = values_x.shape[0] * values_y.shape[0]
    f = np.zeros(array_size, dtype=np.complex128)
    modes_indices = np.zeros((array_size, 2), dtype=np.int)
    for i_x, e_x in enumerate(values_x):
        for i_y, e_y in enumerate(values_y):
            f[i_c] = e_x * e_y
            modes_indices[i_c, :] = (i_x, i_y)
            i_c += 1

    sorted_mode_indices = f.argsort()[::-1]

    return modes_indices[sorted_mode_indices[:], :]

def xCut(input_2d):
    index_x = int(input_2d.shape[0]/2)
    return input_2d[index_x, :]

def yCut(input_2d):
    index_y = int(input_2d.shape[1]/2)
    return input_2d[:, index_y]

def diagonalCutUpwards(input_2d):
    if input_2d.shape[0] < input_2d.shape[1]:
        min_y = input_2d.shape[1]/2 - input_2d.shape[0]/2
        max_y = min_y + input_2d.shape[0]

        work_array = input_2d[:, min_y:max_y]
    elif input_2d.shape[0] > input_2d.shape[1]:
        min_x = input_2d.shape[0]/2 - input_2d.shape[1]/2
        max_x = min_x + input_2d.shape[1]

        work_array = input_2d[min_x:max_x, :]
    else:
        work_array = input_2d

    return np.diag(work_array)

def diagonalCutDownwards(input_2d):
    reverse_array = input_2d[::-1, :]

    return diagonalCutUpwards(reverse_array)


class Enum(object):
    def __init__(self, enum_type):
        self._enum_type = enum_type

    def __eq__(self, candidate):
        return self._enum_type == candidate._enum_type

    def __ne__(self, candidate):
        return not (self==candidate)
