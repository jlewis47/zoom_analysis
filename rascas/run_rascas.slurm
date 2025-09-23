#!/bin/bash
#SBATCH --job-name=rascas
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00

module purge
module load gsl
module load gcc
module load inteloneapi/2024.0
module load mpich


cd /home/jlewis/rascas/f90/test

export OMP_NUM_THREADS=128

#untar

mpiexec -np 8 ./rascas params_rascas.cfg &> rascas.out

#retar