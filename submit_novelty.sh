#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --job-name="novelty_16384"
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=log_novelty.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

# Novelty / memorization analysis. Requires evaluate.py to have run first
# (reads its df_gen_features.csv / df_real_features.csv to exclude degenerate
# generations). For each generated tree: Chamfer to nearest same-species
# training tree, vs the test->train reference and the train->train floor.
python novelty.py \
    --experiment_name finetune-8-512-16384 \
    --data_path data/preprocessed-16384 \
    --n_points 2048 \
    --num_workers 48 \
    --seed 42

# Validation: the retrieval baseline emits training trees, so its gen->train
# distance should collapse to ~0 (the memorization floor). Confirms the method.
python novelty.py \
    --experiment_name retrieval-baseline-16384 \
    --data_path data/preprocessed-16384 \
    --n_points 2048 \
    --num_workers 48 \
    --seed 42
