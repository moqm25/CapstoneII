#!/bin/bash
#############################################################
#           Ansel's WRF-Solar-BNL Installation Script       #
#############################################################
#Installs and performs set-up for WRF-Solar-BNL which is based on WRF-Solar2 built on WRF version 4.1.2
#Intended for and tested on Ubuntu-20.04 via WSL (Windows Subsystem for Linux)


# System type: x86_64-pc-linux-gnu
# OS: Ubuntu-20.04

############################################################
#           Environment Setup                              #
############################################################
# Updating and installing dependencies
sudo apt update && sudo apt upgrade -y 

sudo apt install cpp gcc gfortran -y
#GCC 9.4.0
#CPP 9.4.0
#GFortran 9.4.0

sudo apt install csh perl -y
sudo apt install make m4 -y
sudo apt install python3-pip -y
pip3 install netCDF4 pandas
##############
#Setting Flags
##############

#Location flags
echo "export BASE=`pwd`" >> ~/.bashrc
echo "export DIR=\$BASE/Build_WRF/LIBRARIES" >> ~/.bashrc
echo "export WRF_DIR=\$BASE/Build_WRF/WRF" >> ~/.bashrc
echo "export WPS_DIR=\$BASE/Build_WRF/WPS" >> ~/.bashrc

#Compiler flags
echo "export CC=gcc" >> ~/.bashrc
echo "export CXX=g++" >> ~/.bashrc
echo "export FC=gfortran" >> ~/.bashrc
echo "export FCFLAGS=\"-m64\"" >> ~/.bashrc
echo "export F77=gfortran" >> ~/.bashrc
echo "export FFLAGS=\"-m64\"" >> ~/.bashrc
echo "alias CC=\"gcc\"" >> ~/.bashrc

#Library/Dependency Flags
echo "export LD_LIBRARY_PATH=\$DIR/netcdf/lib:\$DIR/grib2/lib" >> ~/.bashrc
echo "export JASPERLIB=\$DIR/grib2/lib" >> ~/.bashrc
echo "export JASPERINC=\$DIR/grib2/include" >> ~/.bashrc
echo "export NETCDF=\$DIR/netcdf" >> ~/.bashrc
echo "export PATH=\$DIR/netcdf/bin:\$PATH" >> ~/.bashrc
echo "export PATH=\$DIR/mpich/bin:\$PATH" >> ~/.bashrc
echo "export NETCDF_classic=1" >> ~/.bashrc
echo "export LDFLAGS=\"-L\$DIR/netcdf/lib -L\$DIR/grib2/lib\"" >> ~/.bashrc
echo "export CPPFLAGS=\"-I\$DIR/netcdf/include -I\$DIR/grib2/include\"" >> ~/.bashrc

source ~/.bashrc

#Beginning Build
mkdir Build_WRF
cd Build_WRF


#########################################
#       Making Dependent Libraries      #
#########################################

mkdir LIBRARIES
cd LIBRARIES


#NETCDF
export LIBS=""
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/netcdf-c-4.7.2.tar.gz
tar xzf netcdf-c-4.7.2.tar.gz
cd netcdf-c-4.7.2
./configure --prefix=$DIR/netcdf --disable-dap --disable-netcdf-4 --disable-shared
make
sudo make install
cd ..

#ZLIB
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/zlib-1.2.11.tar.gz
tar xzf zlib-1.2.11.tar.gz
cd zlib-1.2.11
./configure --prefix=$DIR/grib2
make
sudo make install
cd ..

#NETCDF-FORTRAN
export LIBS="-lnetcdf -lz"
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/netcdf-fortran-4.5.2.tar.gz
tar xzf netcdf-fortran-4.5.2.tar.gz
cd netcdf-fortran-4.5.2
./configure --prefix=$DIR/netcdf --disable-dap --disable-netcdf-4 --disable-shared
make
sudo make install
cd ..

#MPICH
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/mpich-3.0.4.tar.gz
tar xzf mpich-3.0.4.tar.gz
cd mpich-3.0.4
./configure --prefix=$DIR/mpich -disable-cxx
make
sudo make install
cd ..

#LIBPNG
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/libpng-1.2.50.tar.gz
tar xzf libpng-1.2.50.tar.gz
cd libpng-1.2.50
./configure --prefix=$DIR/grib2
make
sudo make install
cd ..

#JASPER
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/jasper-1.900.1.tar.gz
tar xzf jasper-1.900.1.tar.gz
cd jasper-1.900.1
./configure --prefix=$DIR/grib2
make
sudo make install
cd ..

#############################
# WRF/WPS/GEOG Installation #
#############################
#Getting WRF/WPS
cd ..
git clone https://github.com/levinzx/wrfsolar_bnl_mp.git
mv wrfsolar_bnl_mp/* ./
echo y | rm -r wrfsolar_bnl_mp
rm -r wrfsolar-crm
rm -r wrfsolar-original
mv wrfsolar-bnl/* ./
echo y | rm -r wrfsolar-bnl
mv wrfv4 WRF
mv v41wps WPS

# Getting Base WPS-GEOG Files #
wget https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_high_res_mandatory.tar.gz -O geog_high_res_mandatory.tar.gz
tar -xzf geog_high_res_mandatory.tar.gz


#Getting extra GEOG files Neccesary for Aerosolized modeling
cp -r $BASE/GEOG_Input/* WPS_GEOG

#Setup WRF
cd WRF
./clean
( echo 34 ; echo 1 ) | ./configure
echo "Compiling em_real -- logging in compile.em_real.log -- get a coffee this may take a while..."
./compile em_real>&compile.em_real.log
cd run
ln -sf $BASE/stratocumulous_case/WRF_Input/input_aer_aod_d01.nc .
ln -sf $BASE/stratocumulous_case/WRF_Input/namelist.input .
ln -sf $BASE/stratocumulous_case/WRF_Input/wrfbdy_d01 .
ln -sf $BASE/stratocumulous_case/WRF_Input/wrfinput_d01 .
ln -sf $BASE/stratocumulous_case/WRF_Input/wrfinput_d02 .
cd ../..

#Setup WPS
cd WPS

echo 3 | ./configure
./compile

#############
# Running WPS
###########

#ln -sf $BASE/WPS_Input/namelist.wps .

#  #Geogrid#  #
#./geogrid.exe

#  #Ungrib#  #
#./link_grib.csh $BASE/WPS_Input/fnl*
#./ungrib.exe

#  #Metgrid#  #
#./metgrid.exe


cd ../WRF/run
#ln -sf $WPS_DIR/met_em* .
#####################
#	End						#
##########################################################
echo "Installation has completed"
echo "You can now run mpirun -np <Number of proccessors reccomended=6> ./wrf.exe"
exec bash
exit