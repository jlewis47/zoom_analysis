#!/bin/sh
#PBS -S /bin/sh
#PBS -N rlc
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=24:00:00


module load intelpython

export PYTHONPATH=$PYTHONPATH:/home/jlewis/

cd $PBS_O_WORKDIR

python -u visu_big_picture.py > log_big_pic
