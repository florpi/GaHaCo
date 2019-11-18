#!/bin/bash -l

#SBATCH --ntasks 64 
#SBATCH -J gahaco
#SBATCH -o logs/gahaco.out
#SBATCH -e logs/gahaco.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 2400
#SBATCH --exclusive


export XDG_RUNTIME_DIR=/cosma/home/dp004/dc-cues1/newdir/
export QT_QPA_PLATFORM='offscreen'

python train_model.py --np $SLURM_NTASKS --upload=False --n_splits=4 --models_dir=None
