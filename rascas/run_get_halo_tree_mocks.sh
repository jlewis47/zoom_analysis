#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00

module purge
module load intelpython

# source /home/jlewis/.bashrc

# 
# conda activate 

cd /home/jlewis/zoom_analysis/rascas

export PYTHONPATH=$PYTHONPATH:'/home/jlewis'

# export OMP_NUM_THREADS=128

which python


python -u rascas_halo_tree_mock.py
