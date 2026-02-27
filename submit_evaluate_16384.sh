#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="eval_16384"
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=log_evaluate_16384.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python evaluate_v3.py \
    --experiment_name finetune-8-512-16384 \
    --data_path data/preprocessed-16384 \
    --num_workers 48 \
    --seed 42