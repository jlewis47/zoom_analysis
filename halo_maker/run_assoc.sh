#!/bin/sh
#PBS -S /bin/sh
#PBS -N joe_assoc
#PBS -j oe
#PBS -l nodes=1:ppn=64,walltime=06:00:00

source /home/jlewis/.bashrc

module purge
module load intelpython/3-2024.1.0
module load mpich
module load openmpi

export PYTHONPATH=$PYTHONPATH:'/home/jlewis'


# export OMP_NUM_THREADS=64

date

cd /home/jlewis/zoom_analysis/halo_maker


# mpiexec -np 64 python -u assoc.py >log_assoc
python -u assoc.py >log_assoc

date

exit 0
