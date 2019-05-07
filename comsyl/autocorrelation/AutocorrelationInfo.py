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



import os
import binascii

from time import localtime, strftime, mktime
from datetime import datetime
from socket import gethostname
try:
    import mpi4py.MPI as mpi
except:
    pass
from comsyl.utils.Logger import getTotalLog

# before 1.3: multiply by beam energy stepwidth for energy spreaded calculations
AF_VERSION = 1.3
TIME_FORMAT = "%a, %d %b %Y %H:%M:%S"

class AutocorrelationInfo(object):
    def __init__(self):
        self.__dict = dict()
        self.setTotalSpatialModeIntensity(None)

    def asDictionary(self):
        return self.__dict

    def set(self, key, value):
        self.__dict[key] = value

    def get(self, key):
        return self.__dict[key]

    def keys(self):
        return self.__dict.keys()

    def setConfiguration(self, configuration):
        self.set("configuration", configuration.toString())

    def configuration(self):
        return self.get("configuration")

    def setTag(self, tag):
        self.set("tag", tag)

    def tag(self):
        return self.get("tag")

    def _setHostname(self):
        self.set("hostname", gethostname())

    def hostname(self):
        return self.get("hostname")

    def _setNumberCores(self):
        number_cores = mpi.COMM_WORLD.Get_size()
        self.set("number_cores", str(number_cores))

    def numberCores(self):
        return int(self.get("number_cores"))

    def _setStartTime(self):
        time_string = strftime(TIME_FORMAT, localtime())
        self.set("start_time", time_string)

    def startTime(self):
        return self.get("start_time")

    def logStart(self):
        self._setVersion(AF_VERSION)
        self._setUid()
        self._setHostname()
        self._setNumberCores()
        self._setStartTime()

    def setEndTime(self):
        self.setLog()
        time_string = strftime(TIME_FORMAT, localtime())
        self.set("end_time", time_string)

    def endTime(self):
        return self.get("end_time")

    def differenceTime(self):
        start_time = datetime.strptime(self.startTime(), TIME_FORMAT)
        end_time = datetime.strptime(self.endTime(), TIME_FORMAT)

        difference_time = end_time - start_time

        return str(difference_time)

    def setLog(self):
        self.set("log", getTotalLog())

    def log(self):
        return self.get("log")

    def setTotalSpatialModeIntensity(self, total_spatial_mode_intensity):
        self.set("total_spatial_mode_intensity", total_spatial_mode_intensity)

    def totalSpatialModeIntensity(self):
        return self.get("total_spatial_mode_intensity")

    def version(self):
        return self.get("version")

    def _setVersion(self, version):
        self.set("version", str(version))

    def uid(self):
        return self.get("uid")

    def _setUid(self):
        self.set("uid", str(binascii.hexlify(os.urandom(16)).decode('ascii')))

    def setSourcePosition(self, source_position):
        self.set("source_position", source_position)

    def sourcePosition(self):
        if "source_position" in self.keys():
            return self.get("source_position")

        #TODO: remove
        source_position_log = [i for i in  self.log().split("\n") if "source position" in i][0]

        if "entrance" in source_position_log:
            source_position = "entrance"
        elif "center" in source_position_log:
            source_position = "center"
        else:
            raise Exception("Source position not saved and not found in log")

        return source_position


    @staticmethod
    def stringSeperator():
        return "<<@@>>"

    def toString(self):

        string = ""
        for key, value in self.asDictionary().items():
            string+="%s%s%s%s" %(key, AutocorrelationInfo.stringSeperator(),
                                 value, AutocorrelationInfo.stringSeperator())

        return string

    @staticmethod
    def fromString(string):
        splited_string = string.split(AutocorrelationInfo.stringSeperator())

        info = AutocorrelationInfo()
        for key, value in zip(splited_string[::2], splited_string[1::2]):
            info.set(key, value)

        return info