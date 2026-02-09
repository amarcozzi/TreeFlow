#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="preprocess_4096"
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --output=log_preprocess_4096.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

# Preprocess LAZ files to Zarr format (min 4096 points)
# Output: data/preprocessed-4096/
python preprocess_laz.py \
    --data_path ./FOR-species20K \
    --output_path ./data/preprocessed-4096 \
    --min_points 4096 \
    --num_workers 40
