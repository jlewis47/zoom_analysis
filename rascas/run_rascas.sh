#!/bin/sh
#PBS -S /bin/sh
#PBS -N rlc
#PBS -j oe
#PBS -l nodes=1:ppn=128:has1024gbram,walltime=24:00:00



module purge
module load gsl
module load gcc
module load inteloneapi/2024.0
module load mpich


cd /home/jlewis/rascas/f90/test

export OMP_NUM_THREADS=128

mpiexec -np 8 ./rascas params_rascas.cfg &> rascas.out


exit 0
