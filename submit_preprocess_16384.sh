#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="preprocess_16384"
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --output=log_preprocess_16384.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

# Preprocess LAZ files to Zarr format (min 16384 points)
# Output: data/preprocessed-16384/
python preprocess_laz.py \
    --data_path ./FOR-species20K \
    --output_path ./data/preprocessed-16384 \
    --min_points 16384 \
    --num_workers 40
