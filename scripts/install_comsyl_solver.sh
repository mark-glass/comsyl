#!/bin/bash

#===============================================================================
#
# Script to install COMSYL on ubuntu 18.04 cluster
#
# mpi4py needs:
# sudo apt install lam4-dev libmpich-dev  libopenmpi-dev
#===============================================================================
#
#


# be sure to have oasys installed and evironment activated
# the oasys python should be available under "python"

source /home/manuel/OASYS1.2/define_environment.sh


# clean old stuff
echo "Cleaning old installation files..."
rm -rf =* comsyl petsc* slepc* define_env.sh 


#
# comsyl and dependencies, including mpi4py, are installed with pip
#
python -m pip install comsyl --upgrade

export COMSYL_HOME=`pwd`

#
# PETSc
#

echo "Installing PETSc"
wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.6.4.tar.gz
tar xvfz petsc-3.6.4.tar.gz

mv petsc-3.6.4/ petsc
cd petsc
echo "Set PETSc environment variables! Maybe to .bashrc"
export PETSC_DIR=`pwd`
export PETSC_ARCH=arch-linux2-c-opt

python2 ./configure --with-scalar-type=complex --with-debug=0 COPTFLAGS='-O3 -march=native -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' FOPTFLAGS='-O3 -march=native -mtune=native'

# see at the end of configure, for me: 
make PETSC_DIR=$COMSYL_HOME/petsc PETSC_ARCH=arch-linux2-c-opt all
cd ..

#
# SLEPc
#

echo "Installing SLEPc"


# PETSC_DIR must be diefined
wget http://slepc.upv.es/download/distrib/slepc-3.6.3.tar.gz
tar xvfz slepc-3.6.3.tar.gz
mv slepc-3.6.3 slepc
cd slepc
export SLEPC_DIR=`pwd`
python2 ./configure
# copy the command in the last line. For me: 
make SLEPC_DIR=$PWD PETSC_DIR=$COMSYL_HOME/petsc PETSC_ARCH=arch-linux2-c-opt
#make test # failed!!
cd ..



# install petsc4py
wget https://pypi.python.org/packages/91/6f/91e06666eb695e89880b44d0a8f650999e5fef0972745c1e6bd1dd3107d8/petsc4py-3.6.0.tar.gz
tar xvf petsc4py-3.6.0.tar.gz
mv petsc4py-3.6.0/ petsc4py
cd petsc4py
python setup.py build
python -m pip install .
cd ..

# install slepc4py
wget https://pypi.python.org/packages/bd/ca/50da08d615114b00590de6f6638437eaac7e45a108c76c719ebbd95d82f1/slepc4py-3.6.0.tar.gz
tar xvfz slepc4py-3.6.0.tar.gz
mv slepc4py-3.6.0 slepc4py 
cd slepc4py
python setup.py build
python -m pip install .
cd ..


#
# Create PETSc/SLEPc environment file
# It is a good idea to add the contents of this file to your 
# oasys define_environment.sh and start_oasys.sh
#

echo "#!/bin/bash" > define_env.sh
echo "export PETSC_DIR="$PETSC_DIR >> define_env.sh
echo "export PETSC_ARCH="$PETSC_ARCH >> define_env.sh
echo "export SLEPC_DIR="$SLEPC_DIR >> define_env.sh
chmod 777 define_env.sh
cat define_env.sh


echo "All done. "

#
# test
#
#git clone https://github.com/mark-glass/comsyl
#cd comsyl/comsyl
#echo no | python calculateAutocorrelation.py configurations/septest_cm_new_u18_2m_1h.json

wget https://raw.githubusercontent.com/mark-glass/comsyl/oasys/comsyl/configurations/septest_cm_new_u18_2m_1h.json
mkdir calculations
echo no | python -m comsyl.calculateAutocorrelation septest_cm_new_u18_2m_1h.json


