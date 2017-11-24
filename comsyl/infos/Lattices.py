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



#from comsyl.autocorrelation.SigmaMatrix import SigmaWaist, SigmaMatrixFromCovariance
from syned.storage_ring.electron_beam import ElectronBeam

lb_Ob = ElectronBeam(energy_in_GeV=6.04,
                     current      =0.2,
                     moment_xx    =(37.4e-6)**2,
                     moment_yy    =(3.5e-6)**2,
                     moment_xpxp  =(106.9e-6)**2,
                     moment_ypyp  =(1.2e-6)**2,
                     moment_xxp   =0.0,
                     moment_yyp   =0.0,
                     energy_spread=1.06e-03)


hb_Ob = ElectronBeam(energy_in_GeV=6.04,
                     current      =0.2,
                     moment_xx    =(387.8e-6)**2,
                     moment_yy    =(3.5e-6)**2,
                     moment_xpxp  =(10.3e-6)**2,
                     moment_ypyp  =(1.2e-6)**2,
                     moment_xxp   =0.0,
                     moment_yyp   =0.0,
                     energy_spread=1.06e-03)

correct_new_Ob = ElectronBeam(energy_in_GeV=6.04,
                     current      =0.2,
                     moment_xx    =(27.2e-6)**2,
                     moment_yy    =(3.4e-6)**2,
                     moment_xpxp  =(5.2e-6)**2,
                     moment_ypyp  =(1.4e-6)**2,
                     moment_xxp   =0.0,
                     moment_yyp   =0.0,
                     energy_spread=0.95e-03)

correct_new_Ob_alpha = ElectronBeam(energy_in_GeV=6.04,
                     current      =0.2,
                     moment_xx    =9.38000281183e-10,
                     moment_yy    =1.42697722687e-11,
                     moment_xpxp  =1.942035404e-11,
                     moment_ypyp  =1.88728844475e-12,
                     moment_xxp   =-1.4299803854e-11,
                     moment_yyp   =-1.38966769839e-12,
                     energy_spread=0.95e-03)


new_Ob = ElectronBeam(energy_in_GeV=6.04,
                     current      =0.2,
                     moment_xx    =(27.2e-6)**2,
                     moment_yy    =(3.4e-6)**2,
                     moment_xpxp  =(5.2e-6)**2,
                     moment_ypyp  =(1.4e-6)**2,
                     moment_xxp   =0.0,
                     moment_yyp   =0.0,
                     energy_spread=0.89e-03)


# like newOb but E=6.0
ebs_Ob = ElectronBeam(energy_in_GeV=6.0,
                     current      =0.2,
                     moment_xx    =(27.2e-6)**2,
                     moment_yy    =(3.4e-6)**2,
                     moment_xpxp  =(5.2e-6)**2,
                     moment_ypyp  =(1.4e-6)**2,
                     moment_xxp   =0.0,
                     moment_yyp   =0.0,
                     energy_spread=0.89e-03)


# like newOb but E=6.0
ebs_S28D = ElectronBeam(energy_in_GeV=6.0,
                     current      =0.2,
                     moment_xx    =(30.2e-6)**2,
                     moment_yy    =(3.64e-6)**2,
                     moment_xpxp  =(4.37e-6)**2,
                     moment_ypyp  =(1.37e-6)**2,
                     moment_xxp   =0.0,
                     moment_yyp   =0.0,
                     energy_spread=0.89e-03)


alba = ElectronBeam(energy_in_GeV=3.0,
                     current      =0.2,
                     moment_xx    =(127e-6)**2,
                     moment_yy    =(5.2e-6)**2,
                     moment_xpxp  =(36.2e-6)**2,
                     moment_ypyp  =(3.5e-6)**2,
                     moment_xxp   =0.0,
                     moment_yyp   =0.0,
                     energy_spread=1e-03)

#
#
# round_new_Ob = SigmaWaist(sigma_x=0.275*27.2e-6,
#                             sigma_y=4*3.4e-6,
#                             sigma_x_prime=0.275*5.2e-6,
#                             sigma_y_prime=4*1.4e-6,
#                             sigma_dd=0.95e-03)
#
# new_Ob_alpha = SigmaMatrixFromCovariance(xx=9.38000281183e-10,
#                                          yy=1.42697722687e-11,
#                                          xpxp=1.942035404e-11,
#                                          ypyp=1.88728844475e-12,
#                                          sigma_dd=0.89e-03,
#                                          xxp=-1.4299803854e-11,
#                                          yyp=-1.38966769839e-12)
#
#
# new_Ob_alpha_red = SigmaMatrixFromCovariance(xx=9.38000281183e-10,
#                                              yy=1.42697722687e-11,
#                                              xpxp=1.942035404e-11,
#                                              ypyp=1.88728844475e-12,
#                                              sigma_dd=0.89e-03,
#                                              xxp=-1.4299803854e-11 * 10**-3,
#                                              yyp=-1.38966769839e-12 * 10**-3)
#
# new_Ob_nd = SigmaWaist(sigma_x=27.2e-6,
#                        sigma_y=3.4e-6,
#                        sigma_x_prime=1e-10,
#                        sigma_y_prime=1e-10,
#                        sigma_dd=0.89e-03)
#
# delta_Ob = SigmaWaist(sigma_x=1e-10,
#                       sigma_y=1e-10,
#                       sigma_x_prime=1e-10,
#                       sigma_y_prime=1e-10,
#                       sigma_dd=1e-06)
#
# dream_Ob_01 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 0.1,
#                          sigma_y=3.4e-6 * 0.05 * 0.1,
#                          sigma_x_prime=5.2e-6 * 0.05 * 0.1,
#                          sigma_y_prime=1.4e-6 * 0.05 * 0.1,
#                          sigma_dd=0.89e-03 * 0.05 * 0.1)
#
# dream_Ob_05 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 0.5,
#                          sigma_y=3.4e-6 * 0.05 * 0.5,
#                          sigma_x_prime=5.2e-6 * 0.05 * 0.5,
#                          sigma_y_prime=1.4e-6 * 0.05 * 0.5,
#                          sigma_dd=0.89e-03 * 0.05 * 0.5)
#
# dream_Ob_1 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 1,
#                         sigma_y=3.4e-6 * 0.05 * 1,
#                         sigma_x_prime=5.2e-6 * 0.05 * 1,
#                         sigma_y_prime=1.4e-6 * 0.05 * 1,
#                         sigma_dd=0.89e-03 * 0.05 * 1)
#
# dream_Ob_4 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 4,
#                         sigma_y=3.4e-6 * 0.05 * 4,
#                         sigma_x_prime=5.2e-6 * 0.05 * 4,
#                         sigma_y_prime=1.4e-6 * 0.05 * 4,
#                         sigma_dd=0.89e-03 * 0.05 * 4)
#
# dream_Ob_8 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 8,
#                         sigma_y=3.4e-6 * 0.05 * 8,
#                         sigma_x_prime=5.2e-6 * 0.05 * 8,
#                         sigma_y_prime=1.4e-6 * 0.05 * 8,
#                         sigma_dd=0.89e-03 * 0.05 * 8)
#
# dream_Ob_12 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 12,
#                          sigma_y=3.4e-6 * 0.05 * 12,
#                          sigma_x_prime=5.2e-6 * 0.05 * 12,
#                          sigma_y_prime=1.4e-6 * 0.05 * 12,
#                          sigma_dd=0.89e-03 * 0.05 * 12)
#
# dream_Ob_16 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 16,
#                          sigma_y=3.4e-6 * 0.05 * 16,
#                          sigma_x_prime=5.2e-6 * 0.05 * 16,
#                          sigma_y_prime=1.4e-6 * 0.05 * 16,
#                          sigma_dd=0.89e-03 * 0.05 * 16)
#
# dream_Ob_20 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 20,
#                          sigma_y=3.4e-6 * 0.05 * 20,
#                          sigma_x_prime=5.2e-6 * 0.05 * 20,
#                          sigma_y_prime=1.4e-6 * 0.05 * 20,
#                          sigma_dd=0.89e-03 * 0.05 * 20)
#
# dream_Ob_24 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 24,
#                          sigma_y=3.4e-6 * 0.05 * 24,
#                          sigma_x_prime=5.2e-6 * 0.05 * 24,
#                          sigma_y_prime=1.4e-6 * 0.05 * 24,
#                          sigma_dd=0.89e-03 * 0.05 * 24)
#
# dream_Ob_28 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 28,
#                          sigma_y=3.4e-6 * 0.05 * 28,
#                          sigma_x_prime=5.2e-6 * 0.05 * 28,
#                          sigma_y_prime=1.4e-6 * 0.05 * 28,
#                          sigma_dd=0.89e-03 * 0.05 * 28)
#
# dream_trans_Ob_01 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 0.1,
#                          sigma_y=3.4e-6 * 0.05 * 0.1,
#                          sigma_x_prime=5.2e-6 * 0.05 * 0.1,
#                          sigma_y_prime=1.4e-6 * 0.05 * 0.1,
#                          sigma_dd=0.89e-03)
#
# dream_trans_Ob_05 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 0.5,
#                          sigma_y=3.4e-6 * 0.05 * 0.5,
#                          sigma_x_prime=5.2e-6 * 0.05 * 0.5,
#                          sigma_y_prime=1.4e-6 * 0.05 * 0.5,
#                          sigma_dd=0.89e-03)
#
# dream_trans_Ob_1 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 1,
#                         sigma_y=3.4e-6 * 0.05 * 1,
#                         sigma_x_prime=5.2e-6 * 0.05 * 1,
#                         sigma_y_prime=1.4e-6 * 0.05 * 1,
#                         sigma_dd=0.89e-03)
#
# dream_trans_Ob_4 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 4,
#                         sigma_y=3.4e-6 * 0.05 * 4,
#                         sigma_x_prime=5.2e-6 * 0.05 * 4,
#                         sigma_y_prime=1.4e-6 * 0.05 * 4,
#                         sigma_dd=0.89e-03)
#
# dream_trans_Ob_8 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 8,
#                         sigma_y=3.4e-6 * 0.05 * 8,
#                         sigma_x_prime=5.2e-6 * 0.05 * 8,
#                         sigma_y_prime=1.4e-6 * 0.05 * 8,
#                         sigma_dd=0.89e-03)
#
# dream_trans_Ob_12 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 12,
#                          sigma_y=3.4e-6 * 0.05 * 12,
#                          sigma_x_prime=5.2e-6 * 0.05 * 12,
#                          sigma_y_prime=1.4e-6 * 0.05 * 12,
#                          sigma_dd=0.89e-03)
#
# dream_trans_Ob_16 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 16,
#                          sigma_y=3.4e-6 * 0.05 * 16,
#                          sigma_x_prime=5.2e-6 * 0.05 * 16,
#                          sigma_y_prime=1.4e-6 * 0.05 * 16,
#                          sigma_dd=0.89e-03)
#
# dream_trans_Ob_20 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 20,
#                          sigma_y=3.4e-6 * 0.05 * 20,
#                          sigma_x_prime=5.2e-6 * 0.05 * 20,
#                          sigma_y_prime=1.4e-6 * 0.05 * 20,
#                          sigma_dd=0.89e-03)
#
# dream_trans_Ob_24 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 24,
#                          sigma_y=3.4e-6 * 0.05 * 24,
#                          sigma_x_prime=5.2e-6 * 0.05 * 24,
#                          sigma_y_prime=1.4e-6 * 0.05 * 24,
#                          sigma_dd=0.89e-03)
#
# dream_trans_Ob_28 = SigmaWaist(sigma_x=27.2e-6 * 0.05 * 28,
#                          sigma_y=3.4e-6 * 0.05 * 28,
#                          sigma_x_prime=5.2e-6 * 0.05 * 28,
#                          sigma_y_prime=1.4e-6 * 0.05 * 28,
#                          sigma_dd=0.89e-03)
#
#
# example10_ob= SigmaWaist(sigma_x=33.3317e-06,
#                          sigma_y=2.91204e-06,
#                          sigma_x_prime=16.5008e-06,
#                          sigma_y_prime=2.74721e-06)
#
# new_Ob_es02 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 0.2)
#
# new_Ob_es04 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 0.4)
#
# new_Ob_es06 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 0.6)
#
# new_Ob_es08 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 0.8)
#
# new_Ob_es10 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 1.0)
#
# new_Ob_es12 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 1.2)
#
# new_Ob_es14 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 1.4)
#
# new_Ob_es16 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 1.6)
#
# new_Ob_es18 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 1.8)
#
# new_Ob_es20 = SigmaWaist(sigma_x=27.2e-6,
#                     sigma_y=3.4e-6,
#                     sigma_x_prime=5.2e-6,
#                     sigma_y_prime=1.4e-6,
#                     sigma_dd=0.95e-03 * 2.0)


def latticeByName(name):
    lattices = {"low_beta": lb_Ob,
                "high_beta": hb_Ob,
                "new": new_Ob,
                "ebs_Ob": ebs_Ob,
                "ebs_S28D": ebs_S28D,
                "correct_new": correct_new_Ob,
                "alba": alba,
                # "round_new": round_new_Ob,
                # "new_alpha":new_Ob_alpha,
                # "correct_new_alpha":correct_new_Ob_alpha,
                # "new_alpha_red":new_Ob_alpha_red,
                # "new_nd": new_Ob_nd,
                # "dream01": dream_Ob_01,
                # "dream05": dream_Ob_05,
                # "dream1": dream_Ob_1,
                # "dream4": dream_Ob_4,
                # "dream8": dream_Ob_8,
                # "dream12": dream_Ob_12,
                # "dream16": dream_Ob_16,
                # "dream20": dream_Ob_20,
                # "dream24": dream_Ob_24,
                # "dream28": dream_Ob_28,
                # "dream_trans01": dream_trans_Ob_01,
                # "dream_trans05": dream_trans_Ob_05,
                # "dream_trans1": dream_trans_Ob_1,
                # "dream_trans4": dream_trans_Ob_4,
                # "dream_trans8": dream_trans_Ob_8,
                # "dream_trans12": dream_trans_Ob_12,
                # "dream_trans16": dream_trans_Ob_16,
                # "dream_trans20": dream_trans_Ob_20,
                # "dream_trans24": dream_trans_Ob_24,
                # "dream_trans28": dream_trans_Ob_28,
                # "example10": example10_ob,
                # "new_Ob_nd": new_Ob_nd,
                # "delta": delta_Ob,
                # "new_es02": new_Ob_es02,
                # "new_es04": new_Ob_es04,
                # "new_es06": new_Ob_es06,
                # "new_es08": new_Ob_es08,
                # "new_es10": new_Ob_es10,
                # "new_es12": new_Ob_es12,
                # "new_es14": new_Ob_es14,
                # "new_es16": new_Ob_es16,
                # "new_es18": new_Ob_es18,
                # "new_es20": new_Ob_es20,
               }

    return lattices[name]


