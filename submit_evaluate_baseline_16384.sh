#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="eval_retrieval_baseline"
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=log_evaluate_baseline.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

# Score the retrieval baseline with the identical pipeline used for TreeFlow,
# including the supplementary COV/MMD/1-NNA population metrics (--cov_mmd).
python evaluate.py \
    --experiment_name retrieval-baseline-16384 \
    --data_path data/preprocessed-16384 \
    --max_points 16384 \
    --num_workers 48 \
    --seed 42 \
    --cov_mmd
