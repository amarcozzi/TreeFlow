#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="preprocess_8192"
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --output=log_preprocess_8192.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

# Preprocess LAZ files to Zarr format (min 8192 points)
# Output: data/preprocessed-8192/
python preprocess_laz.py \
    --data_path ./FOR-species20K \
    --output_path ./data/preprocessed-8192 \
    --min_points 8192 \
    --num_workers 40
