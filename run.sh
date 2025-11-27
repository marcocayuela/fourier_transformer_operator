#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=01:00:00
#SBATCH -J kolmogorov90
#SBATCH -o $SCRACTH/logs/kolmogorov90.out

module purge
module load python/3.11  

source activate fto
pip install -r requirements.txt

rsync -av $STORE/datasets $SCRATCH

export DATA_DIR=$SCRATCH/
export LOG_DIR=$SCRATCH/

python main.py 
source deactivate

