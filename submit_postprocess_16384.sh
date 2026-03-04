#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="postprocess_16"
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --output=log_postprocess_16.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python postprocess_samples.py \
    --experiment_name finetune-8-512-16384
