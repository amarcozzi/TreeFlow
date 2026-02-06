#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:4
#SBATCH --job-name="train_dit_16384"
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=2-0
#SBATCH --output=log_train_dit_8_256_16384.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

accelerate launch --num_processes=4 train.py \
    --output_dir experiments \
    --model_type dit \
    --experiment_name "dit-8-256-16384" \
    --data_path ./data/preprocessed-16384 \
    --model_dim 256 \
    --num_heads 8 \
    --num_layers 8 \
    --dropout 0.1 \
    --batch_size 16 \
    --visualize_every 10 \
    --save_every 100 \
    --rotation_augment \
    --shuffle_augment \
    --num_workers 4 \
    --num_epochs 10000 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_scheduler cosine \
    --grad_clip_norm 1.0 \
    --mixed_precision bf16 \
    --cfg_dropout_prob 0.15 \
    --max_points 16384 \
    --sample_exponent 0.3
