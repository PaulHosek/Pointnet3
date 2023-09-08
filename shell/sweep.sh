#!/bin/bash
# call with srun

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=04:00:00

#Loading modules
module load 2022
module load Miniconda3/4.12.0
module load CUDA/11.7.0


source /sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate pointcloud
wandb login




output_log="$HOME/GeometricDL/output_dir/output.log"
error_log="$HOME/GeometricDL/output_dir/error.log"

#Copy input file to scratch
mkdir -p /scratch-shared/$USER/ModelNet10
cp -r $HOME/ModelNet10 /scratch-shared/$USER/ModelNet10
cp -r $HOME/GeometricDL/* /scratch-shared/$USER/GeometricDL
mkdir /scratch-shared/$USER/output_dir




python sweep.py -i /scratch-shared/$USER/ModelNet10
cp /scratch-shared/$USER/output_dir/cloud_size.txt $HOME/GeometricDL/cloud_size.txt
