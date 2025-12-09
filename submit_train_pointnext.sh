#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="pn_64"
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=2-0
#SBATCH --output=log_train_pointnext_64.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

python train.py \
    --output_dir experiments \
    --experiment_name "pointnext-64-0.2" \
    --model_type pointnext \
    --data_path FOR-species20K \
    --csv_path FOR-species20K/tree_metadata_dev.csv \
    --preprocessed_version raw \
    --model_dim 64 \
    --dropout 0.1 \
    --batch_size 64 \
    --visualize_every 5 \
    --sample_exponent 0.3 \
    --rotation_augment \
    --shuffle_augment \
    --num_workers 24 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --use_amp \
    --cfg_dropout_prob 0.1 \
