#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="eval_4096"
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=log_evaluate_4096.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python evaluate_v2.py \
    --experiment_name transformer-8-512-4096 \
    --data_path data/preprocessed-4096 \
    --num_workers 48 \
    --seed 42
