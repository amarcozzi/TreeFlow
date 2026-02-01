#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="preprocess_trees"
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --output=log_preprocess_full.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

# Preprocess LAZ files to Zarr format (all points)
# Output: data/preprocessed-full/
python preprocess_laz.py \
    --data_path ./FOR-species20K \
    --output_path ./data/preprocessed-full \
    --num_workers 40
