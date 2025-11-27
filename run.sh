#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=01:00:00
#SBATCH -J kolmogorov90

module purge
module load python/3.11  

source activate fto

rsync -av $STORE/datasets/projectA $SCRATCH/projectA/

export DATA_DIR=$SCRATCH/
export LOG_DIR=$SCRATCH/

python main.py 
conda deactivate

