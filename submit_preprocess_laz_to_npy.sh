#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="preprocess_trees"
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --output=log_preprocess.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

# Preprocess with multiple voxel resolutions
# This will create:
#   FOR-species20K/npy/raw/
#   FOR-species20K/npy/voxel_0.05m/
#   FOR-species20K/npy/voxel_0.1m/
#   FOR-species20K/npy/voxel_0.2m/

python preprocess_laz_to_npy.py \
    --data_path ./FOR-species20K \
    --output_path ./FOR-species20K/npy \
    --include_raw \
    --voxel_sizes 0.05 0.1 0.2 \
    --splits train test \
    --num_workers 40 \
    --verify \
    --verify_dir voxel_0.1m