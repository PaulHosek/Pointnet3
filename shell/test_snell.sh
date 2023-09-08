#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=fat
#SBATCH --time=00:08:00

#Loading modules
module load 2022
module load Miniconda3/4.12.0
module load CUDA/11.8.0
source /sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate base


output_log="$HOME/GeometricDL/output_dir/output.log"
error_log="$HOME/GeometricDL/output_dir/error.log"

#Copy input file to scratch
cp -r $HOME/GeometricDL/data/ModelNet10 "$TMPDIR"/ModelNet10

#Create output directory on scratch
mkdir "$TMPDIR"/output_dir


echo "now python"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
#python $HOME/GeometricDL/hallo_world.py -i "$TMPDIR"/ModelNet10 -o "$TMPDIR"/output_dir > "$output_log" 2> "$error_log"
python $HOME/GeometricDL/c_pointnet2_classification.py -i "$TMPDIR"/ModelNet10 -o "$TMPDIR"/output_dir > "$output_log" 2> "$error_log"

echo "py done"

#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME/GeometricDL



#python $HOME/GeometricDL/c_pointnet2_classification.py -i "$TMPDIR"/ModelNet10 -o "$TMPDIR"/output_dir


# Debugging tools

#tree "$TMPDIR"/ModelNet10 > "$TMPDIR"/output_dir/ModelNet10_tree_scratch.txt
#python $HOME/GeometricDL/hallo_world.py -i $HOME/GeometricDL/data/ModelNet10/ -o "$TMPDIR"/output_dir
#timeout 2m python $HOME/GeometricDL/c_pointnet2_classification.py -i "$TMPDIR"/ModelNet10 -o "$TMPDIR"/output_dir > "$output_log" 2> "$error_log"

