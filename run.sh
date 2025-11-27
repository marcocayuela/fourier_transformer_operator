#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=01:00:00
#SBATCH -J kolmogorov90
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11  

source activate fto
pip install -r requirements.txt

rsync -av $STORE/data/kolmogorov $SCRATCH

export DATA_DIR=$SCRATCH/
export LOG_DIR=$SCRATCH/

python main.py 
source deactivate

