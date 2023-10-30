#!/bin/bash
#SBATCH --job-name=dense_pc_sweep
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

# load modules
module load 2022
module load Anaconda3/2022.05
module load CUDA/11.7.0

source activate gridifier

# architecture sweep
mkdir -p /scratch-shared/$USER/DensePointClouds
cp -R $HOME/DensePointClouds/* /scratch-shared/$USER/DensePointClouds

cd /scratch-shared/$USER/DensePointClouds

WANDB_CACHE_DIR="$TMPDIR"/wandb_cache python main.py