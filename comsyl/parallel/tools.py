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



from comsyl.waveoptics.Wavefront import NumpyWavefront

def communicateWavefront(self, wavefronts, distribution_plan, global_index, root):
    communicator = distribution_plan.communicator()
    n_rank = distribution_plan.myRank()

    if n_rank != root:
        if global_index in distribution_plan.localRows():
            local_index = distribution_plan.globalToLocalIndex(global_index)
            wavefront = wavefronts[local_index]
            as_array = wavefront.asNumpyArray()

            for i_element, element in enumerate(as_array):
                communicator.send(element, dest=root, tag=i_element)

        return None
    else:
        if global_index in distribution_plan.localRows():
            local_index = distribution_plan.globalToLocalIndex(global_index)
            wavefront = wavefronts[local_index]
        else:
            as_array = list()
            i_source = distribution_plan.rankByGlobalIndex(global_index)
            for i_element in range(3):
                as_array.append(communicator.recv(source=i_source, tag=i_element))
            wavefront = NumpyWavefront.fromNumpyArray(as_array[0],as_array[1],as_array[2])

        return wavefront