#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="gen_retrieval_baseline"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=log_generate_baseline.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

# Nearest-neighbor retrieval-and-scaling baseline.
# For each test tree, retrieve the K nearest-in-height same-species training
# trees and write them to experiments/retrieval-baseline-16384/samples/ in the
# standard sample format. Evaluate afterwards with submit_evaluate_baseline_16384.sh.
python generate_baseline_samples.py \
    --data_path data/preprocessed-16384 \
    --experiment_name retrieval-baseline-16384 \
    --max_points 16384 \
    --num_samples_per_tree 16 \
    --retrieval_pool train \
    --num_workers 16 \
    --seed 42
