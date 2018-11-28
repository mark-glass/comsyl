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
import mpi4py.MPI as mpi

from comsyl.autocorrelation.AutocorrelationBuilderStrategy import AutocorrelationBuilderStrategy
from comsyl.mathcomsyl.utils import trapez2D
from comsyl.utils.Logger import log
from comsyl.parallel.DistributionPlan import DistributionPlan


class BuilderStrategyConvolution(AutocorrelationBuilderStrategy):
    def _postInit(self):
        self._evaluate_r_1_last_x = None
        self._evaluate_r_1_last_y = None

    def evaluate(self, r_1, r_2):
        i_x_2 = self.xIndexByCoordinateFast(r_2[0])
        i_y_2 = self.yIndexByCoordinateFast(r_2[1])

        if self._evaluate_r_1_last_x == r_1[0] and self._evaluate_r_1_last_y == r_1[1]:
            pass
        else:
            self._evaluate_r_1_last_result = self.evaluateAllR_2(r_1)
            self._evaluate_r_1_last_x = r_1[0]
            self._evaluate_r_1_last_y = r_1[1]

        result = self._evaluate_r_1_last_result
        integral = result[i_x_2, i_y_2]

        return integral

    def evaluateAllR_2_Integral(self, r_1):
        i_x_1 = self.xIndexByCoordinate(r_1[0])
        i_y_1 = self.yIndexByCoordinate(r_1[1])

        if self._last_r_1_x == i_x_1:
            if self._last_r_1_y == i_y_1:
                return self._last_result

        result = np.zeros_like(self._field)
        for i_field in range(self.numberFields()):
            self._setActiveField(i_field)

            # Shift to mirrored direction
            self.relativeShiftedCopy(i_x_1, i_y_1, self._field_reverse_conj, self._field_tmp)
            self._field_tmp *= self._rho

            result += self._convolution.convolve2D(self._field, self._field_tmp)

        # it is only dV to first power because the solver normalizes for one integration
        result *= self._grid_area

        self._last_r_1_x = i_x_1
        self._last_r_1_y = i_y_1
        self._last_result = result

        return result

    def evaluateAllR_2(self, r_1):
        result = self.evaluateAllR_2_Integral(r_1) * self._density.staticPartFixedR1(r_1)
        return result

    def evaluateCutX(self):
        log("Calculating horizontal cut")
        i_y = self.yFieldIndexByCoordinate(0.0)

        result = np.zeros((self._field.shape[0], self._field.shape[0]), dtype=np.complex128)

        for r_x in self._field_x_coordinates:
            i_x_1 = self.xFieldIndexByCoordinate(r_x)
            r1 = np.array([r_x, 0.0])
            result[i_x_1, :] = self.evaluateAllR_2(r1)[:, i_y]

        log("done")

        result2 = np.zeros_like(result)
        for x1 in self._field_x_coordinates:
            for x2 in self._field_x_coordinates:
                result2[self.xFieldIndexByCoordinate(x1), self.xFieldIndexByCoordinate(x2)] = result[self.xIndexByCoordinate(x1), self.xIndexByCoordinate(x2)]

        return result2

    def evaluateCutY(self):
        log("Calculating vertical cut")
        i_x = self.xFieldIndexByCoordinate(0.0)

        result = np.zeros((self._field.shape[1], self._field.shape[1]), dtype=np.complex128)

        for r_y in self._field_y_coordinates:
            i_y_1 = self.yFieldIndexByCoordinate(r_y)
            r1 = np.array([0.0, r_y])
            result[i_y_1, :] = self.evaluateAllR_2(r1)[i_x, :]

        log("done")

        result2 = np.zeros_like(result)
        for y1 in self._field_y_coordinates:
            for y2 in self._field_y_coordinates:
                result2[self.yFieldIndexByCoordinate(y1), self.yFieldIndexByCoordinate(y2)] = result[self.yIndexByCoordinate(y1), self.yIndexByCoordinate(y2)]

        return result2

class BuilderStrategyPython(AutocorrelationBuilderStrategy):
    def _postInit(self):
        self._action = 0

        if not self._density.isAlphaZero():
            self._setUpPhases()
        else:
            self._has_phases = False

    def _setUpPhases(self):
        log("Calculating density (integration phases)")

        delta_r = np.array([self._x_coordinates_step_width, self._y_coordinates_step_width])
        frequency = self._density.expPhasePointsPerOscillation((self._field_x_coordinates.max(), self._field_y_coordinates.max()))
        oscillations_per_step = frequency * delta_r


        log("Have %.3f horizontal and %.3f vertical phase oscillations per step" % (oscillations_per_step[0], oscillations_per_step[1]))

        if oscillations_per_step[0] > 0.25:
            raise Exception("ABORT. Need AT LEAST 4 integration points per oscillation. Have %f" % (1/oscillations_per_step[0]))

        if oscillations_per_step[1] > 0.25:
            raise Exception("ABORT. Need AT LEAST 4 integration points per oscillation. Have %f" % (1/oscillations_per_step[1]))


        self._setRhoPhase(self._field_x_coordinates, self._field_y_coordinates)

        yy, xx = np.meshgrid(self._field_y_coordinates, self._field_x_coordinates)

        self._rho_phase_exp_x = np.zeros((len(self._field_x_coordinates), xx.shape[0], xx.shape[1]), dtype=np.complex128)
        self._rho_phase_exp_y = np.zeros((len(self._field_y_coordinates), yy.shape[0], yy.shape[1]), dtype=np.complex128)

        for r_x in self._field_x_coordinates:
            i_x = self.xIndexByCoordinate(r_x)
            rho_phase = self._rho_phase_x[i_x]
            self._rho_phase_exp_x[i_x, :, :] = np.exp(xx * rho_phase)

        for r_y in self._field_y_coordinates:
            i_y = self.xIndexByCoordinate(r_y)
            rho_phase = self._rho_phase_y[i_y]
            self._rho_phase_exp_y[i_y, :, :] = np.exp(yy * rho_phase)

        self._coordinate_map_x = np.zeros(len(self._field_x_coordinates), dtype=np.int)
        self._coordinate_map_y = np.zeros(len(self._field_y_coordinates), dtype=np.int)
        self._coordinate_map_minus_x = np.zeros(len(self._field_x_coordinates), dtype=np.int)
        self._coordinate_map_minus_y = np.zeros(len(self._field_y_coordinates), dtype=np.int)

        for i_r_x, r_x in enumerate(self._field_x_coordinates):
            self._coordinate_map_x[i_r_x] = self.xIndexByCoordinate(-r_x)
            self._coordinate_map_minus_x[i_r_x] = self.xIndexByCoordinate(r_x)

        for i_r_y, r_y in enumerate(self._field_y_coordinates):
            self._coordinate_map_y[i_r_y] = self.yIndexByCoordinate(-r_y)
            self._coordinate_map_minus_y[i_r_y] = self.yIndexByCoordinate(r_y)

        self._has_phases = True

    def calculateRhoPhase(self, dr):
        i_x = self.xIndexByCoordinate(-dr[0])
        i_y = self.yIndexByCoordinate(-dr[1])

        self._rho_phase_tmp[:, :] = self._rho_phase_exp_x[i_x, :, :] * self._rho_phase_exp_y[i_y, :, :]
        return self._rho_phase_tmp

    def evaluateAllR_2_Integral(self, r_1):
        i_x_1 = self.xIndexByCoordinate(-r_1[0])
        i_y_1 = self.yIndexByCoordinate(-r_1[1])

        result = np.zeros_like(self._field)

        e_1 = np.zeros_like(self._field)
        e_2 = np.zeros_like(self._field)

        self.relativeShiftedCopy(i_x_1, i_y_1, self._field_conj, e_1)
        field_product = e_1 * self._rho * self.calculateRhoPhase((-r_1[0], -r_1[1]))

        for i_field in range(self.numberFields()):
            self._setActiveField(i_field)

            for i_r_x, r_x in enumerate(self._field_x_coordinates):
                i_x_2 = self._coordinate_map_x[i_r_x]
                for i_r_y, r_y in enumerate(self._field_y_coordinates):
                    i_y_2 = self._coordinate_map_y[i_r_y]

                    self.relativeShiftedCopy(i_x_2, i_y_2, self._field, e_2)

                    e_2 *= field_product
                    e_2 *= self._rho_phase_exp_x[i_x_2, :, :]
                    e_2 *= self._rho_phase_exp_y[i_y_2, :, :]

                    integral = np.sum(e_2)

                    result[i_r_x, i_r_y] += integral
        
        result *= self._grid_area

        return result

    def evaluateAllR_2_Fredholm_parallel(self, v_in, v_out):
        self._action += 1

        if self.doNotUseConvolutions():
            self.evaluateAllR_2_Fredholm_parallel_direct(v_in, v_out)
        else:
            self.evaluateAllR_2_Fredholm_parallel_convolution(v_in, v_out)

    # TODO: trapez is realtively different around 1e-6 in norm - or not - to be investigated
    def evaluateAllR_2_Fredholm_parallel_direct(self, v_in, v_out):

        if not self._has_phases:
           self._setUpPhases()

        v = v_in.fullData().copy()

        H = np.zeros_like(self._field)
        tmp2 = np.zeros((self._field.shape[0] * self._field.shape[1]), dtype=np.complex128)
        result = np.zeros_like(self._field)
        tmp = np.zeros( (self._field_y_coordinates.shape[0], self._field.shape[0] * self._field.shape[1]), dtype=np.complex128)

        e_1 = np.zeros_like(self._field)
        e_2 = np.zeros_like(self._field)

        distribution_plan = DistributionPlan(communicator=mpi.COMM_WORLD, n_rows=len(self._field_x_coordinates), n_columns=len(self._field_y_coordinates))
        local_rows = distribution_plan.localRows()
        range_y_coordinates = tuple(range(len(self._field_y_coordinates)))

        for i_field in range(self.numberFields()):
            self._setActiveField(i_field)

            for i_r_x in local_rows:
                i_x_1 = self._coordinate_map_x[i_r_x]
                i_x_minus_1 = self._coordinate_map_minus_x[i_r_x]
                for i_r_y in range_y_coordinates:
                    i_y_1 = self._coordinate_map_y[i_r_y]
                    i_y_minus_1 = self._coordinate_map_minus_y[i_r_y]
                    self.relativeShiftedCopy(i_x_1, i_y_1, self._field_conj, e_1)
                    tmp[i_r_y, :] = e_1.ravel()
                    tmp[i_r_y, :] *= self._rho_phase_exp_y[i_y_minus_1, :, :].ravel()

                tmp2[:] = v * self._rho_phase_exp_x[i_x_minus_1, :, :].ravel()
                H[i_r_x, :] = tmp.dot(tmp2)

            v_out.sumFullData(H.ravel())

            field_product = self._rho.flatten() * v_out.fullData()

            for i_r_x in local_rows:
                i_x_2 = self._coordinate_map_x[i_r_x]
                for i_r_y in range_y_coordinates:
                    i_y_2 = self._coordinate_map_y[i_r_y]

                    self.relativeShiftedCopy(i_x_2, i_y_2, self._field, e_2)
                    tmp[i_r_y, :] = e_2.ravel()
                    tmp[i_r_y, :] *= self._rho_phase_exp_y[i_y_2, :, :].ravel()

                tmp2[:] = field_product * self._rho_phase_exp_x[i_x_2, :, :].ravel()
                result[i_r_x, :] += tmp.dot(tmp2)

        # it is only dV to first power because the solver normalizes for one integration
        result *= self._grid_area
        v_out.sumFullData(result.ravel())

    # symmetry_point
    def evaluateAllR_2_Fredholm_parallel_convolution(self, v_in, v_out):
        f = v_in.fullData().reshape(self._x_coordinates.shape[0], self._y_coordinates.shape[0])

        distribution_plan = DistributionPlan(communicator=mpi.COMM_WORLD, n_rows=self.numberFields(), n_columns=self.numberFields())
        local_rows = distribution_plan.localRows()

        res = np.zeros_like(self._field)
        for i_field in local_rows:
            self._setActiveField(i_field)

            scal_prod_action = self._convolution.convolve2D(f, self._field_reverse_conj)
            self._field_tmp[:, :] = scal_prod_action * self._rho
            res += self._convolution.convolve2D(self._field, self._field_tmp)

        res *= self._grid_area
        v_out.sumFullData(res.ravel())

    def evaluateAllR_2_Fredholm_parallel_compare(self, v_in, v_out):

        v_in_clone = v_in.clone()

        self.evaluateAllR_2_Fredholm_parallel_direct(v_in, v_out)
        self.evaluateAllR_2_Fredholm_parallel_convolution(v_in_clone, v_in_clone)

        #from comsyl.math.utils import plotSurface
        # plotSurface(self._field_x_coordinates,self._field_y_coordinates, np.abs(res))
        # plotSurface(self._field_x_coordinates,self._field_y_coordinates, np.abs(v2))
        # plotSurface(self._field_x_coordinates,self._field_y_coordinates, np.abs(v2-res))
        res = v_out.fullData()
        v2 = v_in_clone.fullData()
        log("abs error, rel error: %f, %f"  % (np.linalg.norm(np.abs(res)-np.abs(v2)), np.linalg.norm(np.abs(res-v2)/np.linalg.norm(np.abs(res)))))

    def evaluateAllR_2(self, r_1):
        result = self.evaluateAllR_2_Integral(r_1) * self._density.staticPartFixedR1(r_1)
        return result
