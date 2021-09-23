#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=FUA32_WPJET1
#SBATCH --partition=skl_fua_prod
#SBATCH --time=24:00:00

export PATH=$PATH:$HOME/bin
module load profile/global
module load intel/pe-xe-2018--binary # Needed for numpy
module load mkl/2018--binary #Needed for numpy
module load szip/2.1--gnu--6.1.0 #Needed for hdf5
module load zlib/1.2.8--gnu--6.1.0 #Needed for hdf5
module load intelmpi/2018--binary #Needed for hdf5
module load hdf5/1.8.18--intelmpi--2018--binary
export HDF5_DIR=$HDF5_HOME
module load netcdf/4.6.1--intelmpi--2018--binary
export NETCDF4_DIR=$NETCDF_HOME

module load gnu/7.3.0
module load nag
module load python/3.6.4
module load numpy/1.14.0--python--3.6.4
module load tensorflow/1.6.0--python--3.6.4

module load cmake
export PYTHONPATH=$PYTHONPATH:$PYTHONPATH_modshare
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

cd $SLURM_SUBMIT_DIR

python -c 'from qlknn.training import train_NDNN; train_NDNN.train_NDNN_from_folder()'
