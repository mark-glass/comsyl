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


#
# Note on installation of comsyl: 
#
# comsyl is not fully pip-installable, as it requires libraries for the solver
# that are hard to compile (petsc, slepc, and the python binders petsc4py and
# slepc4py)
# Fot pip-building the oasys interface, these solver-dependencies have been
# removed from the setup.py and should be built independently (see 
# installation instructions).
#
# The comsyl that is pip installable can be used with oasys to load/write
# comsyl files and therefore propagate comsyl results along the beamlines.
#
#
# To create the pip installation: 
# git checkout oasys
# python setup.py sdist
# python -m twine upload dist/*
#




__authors__ = ["M Glass - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"
__date__ = "20/04/2017"



from setuptools import setup

setup(name='comsyl',
    version='1.0.12',
    description='Coherent modes for synchrotron light',
    author='Mark Glass',
    author_email='mark.glass@esrf.fr',
    url='https://github.com/mark-glass/comsyl/',
    packages=['comsyl',
        'comsyl.autocorrelation',
        'comsyl.utils',
        'comsyl.mathcomsyl',
        'comsyl.waveoptics',
        'comsyl.infos',
        'comsyl.parallel',
        ],
    install_requires=[
        'numpy',
        'scipy',
        'syned',
        'wofry',
        'srxraylib',
        'oasys-srwpy',
        'mpi4py',
        'h5py', #  slepc4py', petsc4py', mpi4py',
        ],
    package_data={
        'configurations':["septest_cm_new_u18_2m_1h.json"],
        'calculations':["readme.txt"],
        },
    setup_requires=[
        'setuptools',
        ],
    )
