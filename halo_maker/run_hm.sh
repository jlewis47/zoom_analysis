#!/bin/bash
#SBATCH --job-name=halomaker_id242704
#SBATCH --nodelist=i[02-32]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00

module purge
module load inteloneapi/2022.2

ulimit -s unlimited

export OMP_NUM_THREADS=1

date

cd 

cur_nb=$(printf "%d" $(find . -maxdepth 1 -name 'run_*.log' -type f | cut -d _ -f 2 | cut -d . -f 1 | sort | tail -n 1))
nb=$(printf "%05d" $((cur_nb+1)))

ulimit -s unlimited

./HaloMaker >& run_$nb.log

date

exit 0
