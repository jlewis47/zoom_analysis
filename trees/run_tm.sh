#!/bin/bash
#SBATCH --job-name=treemaker_id242704
##SBATCH --nodelist=i[02-32]
#SBATCH --partition=comp,compl,pscomp,pscompl
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

module purge
module load inteloneapi/2022.2
module load intelpython/3-2024.1.0

export OMP_NUM_THREADS=1
export PYTHONPATH=$PYTHONPATH:/home/jlewis

ulimit -s unlimited

date

cd 

cur_nb=$(printf "%d" $(find . -maxdepth 1 -name 'run_*.log' -type f | cut -d _ -f 2 | cut -d . -f 1 | sort | tail -n 1))
nb=$(printf "%05d" $((cur_nb+1)))


./TreeMaker >& run_$nb.log

./manipulate_mergertree -ftr tree.dat -out tree_rev.dat -rev true >& rev_tree_log

python /home/jlewis/codes/zoom_analysis/trees/make_tree_rev_offsets.py tree_rev.dat . >& rev_tree_offsets_log

date

exit 0
