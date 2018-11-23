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


import time

from comsyl.parallel.utils import isMaster

import sys


class Logger(object):
    def __init__(self):
        self.resetLog()

    def resetLog(self):
        self._total_log = []

    def log(self, log_string):
        if isMaster():
            self.logAll(log_string)

    def logAll(self, log_string):
        output = "[%s] %s" %(time.strftime("%Y-%m-%d %H:%M:%S"), log_string)
        print(output)
        self._total_log.append(output)
        sys.stdout.flush()

    def totalLog(self):
        return "\n".join(self._total_log)

logger = Logger()


def log(log_string):
    logger.log(log_string)

def logAll(log_string):
    logger.logAll(log_string)

def getTotalLog():
    return logger.totalLog()


def resetLog():
    logger.resetLog()


def logProgress(n_max, n_current, process):
    try:
        if n_current%(int(n_max*0.02)) == 0:
            log("%s: %i / %i " % (process, n_current+1, n_max))
    except ZeroDivisionError:
            log("%s: %i" % (process, n_current+1))
