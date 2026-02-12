#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="postprocess_4096"
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --output=log_postprocess_4096.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python postprocess_samples.py \
    --experiment_name transformer-8-512-4096
