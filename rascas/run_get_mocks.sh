#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00

module purge
module load intelpython

cd /home/jlewis/zoom_analysis/rascas

export PYTHONPATH=$PYTHONPATH:'/home/jlewis'

# export OMP_NUM_THREADS=128

python -u rascas_mock.py
