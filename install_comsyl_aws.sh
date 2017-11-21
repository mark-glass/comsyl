#!/bin/bash

#===============================================================================
#
# Script to install COMSYL on an Amazon Web Services (AWS) EC2 ubuntu 16.04 cluster
#
#===============================================================================
#
#


# clean old stuff
echo "Cleaning old installation files..."
rm -rf =* comsyl petsc* slepc* 

#
# mpi4py and other stuff installed with pip
#
sudo apt -y install python3-pip
pip3 install numpy
pip3 install scipy
pip3 install mpi4py

# install srxraylib and syned
pip3 install srxraylib
pip3 install syned


export COMSYL_HOME=`pwd`

#
# PETSc
#

echo "Installing PETSc"
#tar xvfz TarFiles/petsc-3.6.4.tar.gz
wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.6.4.tar.gz
tar xvfz petsc-3.6.4.tar.gz

mv petsc-3.6.4/ petsc
cd petsc
echo "Set PETSc environment variables! Maybe to .bashrc"
export PETSC_DIR=`pwd`
export PETSC_ARCH=arch-linux2-c-opt

python2 ./configure --with-scalar-type=complex --with-debug=0 COPTFLAGS='-O3 -march=native -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' FOPTFLAGS='-O3 -march=native -mtune=native'
#./configure --with-scalar-type=complex --with-debug=0
#./configure --with-cc=/usr/bin/mpicc.openmpi --with-mpi-f90=/usr/bin/mpif90.openmpi --with-mpiexec=/usr/bin/mpiexec.openmpi --with-cxx=/usr/bin/mpicxx.openmpi --with-scalar-type=complex --with-mpi-dir=/usr/lib/openmpi --with-debug=0 COPTFLAGS='-O3 -march=native -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' FOPTFLAGS='-O3 -march=native -mtune=native'

# see at the enc of configure, for me: 
make PETSC_DIR=$COMSYL_HOME/petsc PETSC_ARCH=arch-linux2-c-opt all
cd ..

#
# SLEPc
#

echo "Installing SLEPc"


#git clone https://bitbucket.org/slepc/slepc
#curl -O http://slepc.upv.es/download/download.php?filename=slepc-3.7.3.tar.gz
# PETSC_DIR must be diefined
wget http://slepc.upv.es/download/distrib/slepc-3.6.3.tar.gz
tar xvfz slepc-3.6.3.tar.gz
#tar xvfz  TarFiles/slepc-3.6.3.tar.gz
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
#tar xvf TarFiles/petsc4py-3.6.0.tar.gz
tar xvf petsc4py-3.6.0.tar.gz
mv petsc4py-3.6.0/ petsc4py
cd petsc4py
python3 setup.py build
pip3 install .
cd ..

# install slepc4py
wget https://pypi.python.org/packages/bd/ca/50da08d615114b00590de6f6638437eaac7e45a108c76c719ebbd95d82f1/slepc4py-3.6.0.tar.gz
#tar xvfz TarFiles/slepc4py-3.6.0.tar.gz
tar xvfz slepc4py-3.6.0.tar.gz
mv slepc4py-3.6.0 slepc4py 
cd slepc4py
python3 setup.py build
pip3 install .
cd ..



# comsyl
git clone https://github.com/mark-glass/comsyl
cd comsyl
git checkout syned
python3 setup.py build
python3 setup.py develop --user
cd ..


echo "All done. "


#
# install srw
#

rm -rf SRW
export PYTHON_SITE_PACKAGES=`python3 -c "import site; print(site.getusersitepackages())"`

echo "Cleaning old installation files..."
sudo rm -f $PYTHON_SITE_PACKAGES/srwlpy*.so
sudo rm -f $PYTHON_SITE_PACKAGES/uti*.py 
sudo rm -f $PYTHON_SITE_PACKAGES/srwl_uti*.py 
sudo rm -f $PYTHON_SITE_PACKAGES/srwlib.py

echo "Downloading SRW zip. This takes time"
wget https://github.com/ochubar/SRW/archive/master.zip > /dev/null
unzip master.zip
#unzip TarFiles/SRW-master.zip
mv SRW-master SRW


#
# build fftw
#
cd SRW/ext_lib
rm -rf  fftw-2.1.5
tar -zxvf fftw-2.1.5.tar.gz
cd fftw-2.1.5
./configure --enable-float --with-pic
cp Makefile Makefile.orig
# add -fPIC option to CFLAGS in Makefile
#TODO : this did not work for Mac
sed -e "s/^CFLAGS =/CFLAGS = -fPIC/" Makefile -i
make
cp fftw/.libs/libfftw.a ..
cd ../..
echo "Done fftw"
pwd

#
# build srw
#
cd cpp/gcc
# backup
cp Makefile Makefile.orig
# Modify existing Makefile
mv Makefile Makefile.tmp
# remove existing PYFLAGS and PYPATH
sed -i -e "/^PYFLAGS/d" Makefile.tmp 
sed -i -e "/^PYPATH/d" Makefile.tmp
# add the correct python path, include and lib directories
echo "PYPATH = /usr/" >> Makefile
#TODO make this automatic (add python3.4)
echo "PYFLAGS = -I\$(PYPATH)/include/python3.5 -L\$(PYPATH)/lib/python3.5" >> Makefile
cat Makefile.tmp >> Makefile
rm Makefile.tmp
# make (creates libsrw.a)
rm -f libsrw.a
make -j8 clean lib

# make (creates srwlpy.so and puts it in env/work/srw_python/)
cd ../py
cp Makefile Makefile.orig
#TODO check in mac, but this change seems not to be needed
sed -i -e "s/python setup/python3 setup/" Makefile
sed -i -e "s/gcc\/srwlpy/gcc\/srwlpy\*/" Makefile
make python

cd ../../

#
# install
#
cp env/work/srw_python/srwlpy*.so $PYTHON_SITE_PACKAGES
cp env/work/srw_python/uti*.py $PYTHON_SITE_PACKAGES
cp env/work/srw_python/srwl_uti*.py $PYTHON_SITE_PACKAGES
cp env/work/srw_python/srwlib.py $PYTHON_SITE_PACKAGES
#
cd ..
echo "All done for SRW." 

#
# Create PETSc/SLEPc environment file
#

echo "#!/bin/bash" > define_env.sh
echo "export PETSC_DIR="$PETSC_DIR >> define_env.sh
echo "export PETSC_ARCH="$PETSC_ARCH >> define_env.sh
echo "export SLEPC_DIR="$SLEPC_DIR >> define_env.sh
chmod 777 define_env.sh
cat define_env.sh
