#!/bin/bash
#SBATCH --job-name=untar&halomaker&clean
#SBATCH --nodelist=i[02-32]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

source /home/jlewis/.bashrc

ulimit -s unlimited

module purge
module load intelpython/3-2024.1.0
module load inteloneapi/2022.2


export PYTHONPATH=$PYTHONPATH:'/home/jlewis'

set -e #guaranteed to throw and error and stop if anything fails

cd $SLURM_SUBMIT_DIR

simdir=$1

outdirs=$(find $simdir -maxdepth 1 -type d -name "output*")



#find earliest snap with galaxies
gal_bricks=$(find $simdir"/HaloMaker_stars2_dp_rec_dust/" -maxdepth 1 -type f -name "tree_bricks*")
minbricks=9999
for brick in $gal_bricks
do
    bricknum=$(echo $brick | grep -o "[0-9]*$")
    bricknum_int=$(($(echo $bricknum|bc)+0)) #force to be interger
    if [ $bricknum_int -lt $minbricks ]
    then
        minbricks=$bricknum_int
    fi
done

# echo $minbricks

# echo $outdirs

for outdir in $outdirs
do


    outnum=$(echo $outdir | grep -o "[0-9]*$")
    outnum_int=$(($(echo $outnum|bc)+0)) #force to be interger

    echo "Processing output "$outnum_int

    if [ $outnum_int -lt $minbricks ] #skip outputs before galaxies
    then
        continue
    fi

    #what to do it no existing gal_bricks ? will minbricks be 9999?

    run=0

    # echo $outdir
    # echo $outnum_int

    if test -f $(printf "%s/HaloMaker_DM_dust/tree_bricks%03d\n" "$simdir" "$outnum_int")
    then
        echo "HaloMaker output already exists for output "$outnum_int
    else
        run=$(($run+1))
    fi

    if test -f $(printf "%s/HaloMaker_stars2_dp_rec_dust/tree_bricks%03d\n" "$simdir" "$outnum_int")
    then
        echo "GalaxyMaker output already exists for output "$outnum_int
    else
        run=$(($run+2))
    fi

    # echo $run

    if [ $run -gt 0 ]
    then
    
        if ! test -f $outdir"/part_"$outnum".out00001" || ! test -f $outdir"/amr_"$outnum".out00001"
        then
            #check if I need to untar... is there a part file and an amr file ?
            echo "Untaring output "$outnum_int

            cd $simdir

            loc_tarpath="output_"$outnum".tar"

            echo "Untaring "$loc_tarpath

            #find substring in first file path that if path to outputdir
            first_file=$(tar tvf $loc_tarpath | head -n 1)
            for ifind in {1..10}; do
                test=$(echo $first_file | cut -d ' ' -f $ifind)
                if echo $test | grep -q "output_"$outnum
                then
                    break
                fi
            done
            first_file_path=$(echo $first_file | cut -d ' ' -f $ifind)
            #get number of backslashes in path - then we know how many
            #to strip off when untaring
            num_backslash=$(echo $first_file_path | grep -o "/" | wc -l)


            echo tar xvf $loc_tarpath --strip-components=$((num_backslash-1))
            tar xvf $loc_tarpath --strip-components=$((num_backslash-1))

            cd /home/jlewis/zoom_analysis/halo_maker
        fi

        python -u setup_halo_maker.py -sim_dirs $simdir --nolaunch --snap $outnum_int

        if [ $(($run%2)) -ne 0 ]
        then
            cd $simdir"/HaloMaker_DM_dust"
            echo "Running HaloMaker for output "$outnum_int
            ./HaloMaker
        fi

        if [ $run -gt 1 ]
        then
            cd $simdir"/HaloMaker_stars2_dp_rec_dust"
            echo "Running GalaxyMaker for output "$outnum_int
            ./HaloMaker
        fi

        #now run association if needed !!!
        if ! test -f $simdir"/association/.h5"
        then
            cd /home/jlewis/zoom_analysis/halo_maker
            python -u /home/jlewis/zoom_analysis/halo_maker/assoc.py --sim_dirs $simdir
        fi


        echo "Compressing output "$outnum_int


        if ! test -f $simdir"/halogal_data/halo_data_"$outnum_int".h5"
        then
            cd /home/jlewis/compress_zoom
            python -u /home/jlewis/compress_zoom/compress_zoom.py $simdir --snaps $outnum_int
        fi

        if test -f $simdir"/halogal_data/halo_data_"$outnum_int".h5"
        then
            echo "Cleaning up output "$outnum_int
            find $simdir -mindepth 2 -maxdepth 2 -type f -path *output* ! -name "*.txt" ! -name "*.tar" -exec rm -f {} +;
        fi

        cd $SLURM_SUBMIT_DIR

    fi
done

echo "done"
date
exit 0