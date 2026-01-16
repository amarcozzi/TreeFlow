#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="preprocess_trees"
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --output=log_preprocess_4096.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

python preprocess_laz_to_npy.py \
    --data_path ./FOR-species20K \
    --output_path ./FOR-species20K/preprocessed_4096 \
    --splits train test \
    --min_points 4096 \
    --num_workers 40 \
    --verify
