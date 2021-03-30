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



from syned.storage_ring.magnetic_structures.undulator import Undulator


esrf_undulator_20_2m = Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.1911,
                                 period_length    =0.020,
                                 number_of_periods=100,
                                 )

esrf_undulator_18_4m = Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.68,
                                 period_length    =0.018,
                                 number_of_periods=int(4.0/0.018),
                                 )

esrf_undulator_18_2500mm = Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.851076,
                                 period_length    =0.018,
                                 number_of_periods=int(2.5/0.018),
                                 )


esrf_undulator_18_2m = Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.68,
                                 period_length    =0.018,
                                 number_of_periods=int(2.0/0.018),
                                 )


esrf_undulator_18_1m = Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.68,
                                 period_length    =0.018,
                                 number_of_periods=int(1.0/0.018),
                                 )



short_undulator = Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.68,
                                 period_length    =0.018,
                                 number_of_periods=int(1.0/0.018),
                                 )


esrf_u18_3__1_4m_hb =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =0.445,
                                 period_length    =0.0183,
                                 number_of_periods=int(1.4/0.0183),
                                 )

esrf_u18_3__1_4m_ebs =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =0.411,
                                 period_length    =0.0183,
                                 number_of_periods=int(1.4/0.0183),
                                 )

esrf_undulator_17_2m = Undulator(K_horizontal     =0.0,
                                 K_vertical       =0.4842,
                                 period_length    =0.017,
                                 number_of_periods=int(2.0/0.017),
                                 )

esrf_undulator_42_4m = Undulator(K_horizontal     =0.0,
                                 K_vertical       =2.092,
                                 period_length    =0.042,
                                 number_of_periods=int(4.0/0.042),
                                 )


alba_u21_2m =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.159,
                                 period_length    =0.0213,
                                 number_of_periods=92,
                                 )

elettra_u100_4m =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.92,
                                 period_length    =0.100,
                                 number_of_periods=40,
                                 )


alsu_u38_0p3keV =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.810,
                                 period_length    =0.038,
                                 number_of_periods=54,
                                 )

alsu_u38_0p6keV =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.0,
                                 period_length    =0.038,
                                 number_of_periods=54,
                                 )

alsu_u38_0p8keV =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =0.693,
                                 period_length    =0.038,
                                 number_of_periods=54,
                                 )

alsu_u38_1p7keV =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.224,
                                 period_length    =0.038,
                                 number_of_periods=54,
                                 )

apsu_10keV =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =0.85729,
                                 period_length    =0.025,
                                 number_of_periods=184,
                                 )

# use 5th harmonic for 15 keV
apsu_25keV =  Undulator(K_horizontal     =0.0,
                                 K_vertical       =1.862765,
                                 period_length    =0.025,
                                 number_of_periods=184,
                                 )

def undulatorByName(name):
    undulators = {"esrf_u18_4m": esrf_undulator_18_4m,
                  "esrf_u20_2m": esrf_undulator_20_2m,
                  "esrf_u18_2m": esrf_undulator_18_2m,
                  "esrf_u18_2500mm": esrf_undulator_18_2500mm,
                  "esrf_u18_1m": esrf_undulator_18_1m,
                  "short_undulator": short_undulator,
                  "esrf_u18.3_1.4m": esrf_u18_3__1_4m_hb,
                  "esrf_u18.3_1.4m_ebs": esrf_u18_3__1_4m_ebs,
                  "esrf_u42_4m": esrf_undulator_42_4m,
                  "esrf_u17_2m": esrf_undulator_17_2m,
                  "alba_u21_2m": alba_u21_2m,
                  "elettra_u100_4m": elettra_u100_4m,
                  "alsu_u38_0p3keV": alsu_u38_0p3keV,
                  "alsu_u38_0p6keV": alsu_u38_0p6keV,
                  "alsu_u38_0p8keV": alsu_u38_0p8keV,
                  "alsu_u38_1p7keV": alsu_u38_1p7keV,
                  "apsu_10keV": apsu_10keV,
                  "apsu_25keV": apsu_25keV,
                 }

    return undulators[name]
