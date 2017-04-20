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



from BeamlineComponents.Source.UndulatorVertical import UndulatorVertical

esrf_undulator_18_4m = UndulatorVertical(1.68,
                                      0.018,
                                      int(4.0/0.018),
                                     )

esrf_undulator_18_2m = UndulatorVertical(1.68,
                                         0.018,
                                         int(2.0/0.018),
                                        )

esrf_undulator_18_1m = UndulatorVertical(1.68,
                                         0.018,
                                         int(1.0/0.018),
                                        )

short_undulator = UndulatorVertical(1.68,
                                    0.018,
                                    int(1.0/0.018),
                                   )

esrf_u18_3__1_4m = UndulatorVertical(0.445,
                                     0.0183,
                                     int(1.4/0.0183),
                                     )



def undulatorByName(name):
    undulators = {"esrf_u18_4m": esrf_undulator_18_4m,
                  "esrf_u18_2m": esrf_undulator_18_2m,
                  "esrf_u18_1m": esrf_undulator_18_1m,
                  "short_undulator": short_undulator,
                  "esrf_u18.3_1.4m": esrf_u18_3__1_4m,
                 }

    return undulators[name]
