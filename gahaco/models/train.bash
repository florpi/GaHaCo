#!/bin/bash -l

#SBATCH --ntasks 28
#SBATCH -t 05:00:00
#SBATCH -J gahaco_rnf
#SBATCH -o ../../logs/gahaco_rnf.out
#SBATCH -e ../../logs/gahaco_rnf.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive

export XDG_RUNTIME_DIR=/cosma/home/dp004/dc-beck3/4_GaHaCo/GaHaCo/comet/
export QT_QPA_PLATFORM='offscreen'

module unload python
module load python/3.6.5

python3 main.py 
