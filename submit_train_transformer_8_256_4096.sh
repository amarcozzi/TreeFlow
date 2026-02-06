#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:4
#SBATCH --job-name="tf_8_256_4096"
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=2-0
#SBATCH --output=log_train_transformer_8_256_4096.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

accelerate launch --num_processes=4 train.py \
    --output_dir experiments \
    --model_type transformer \
    --experiment_name "transformer-8-256-4096" \
    --data_path ./data/preprocessed-4096 \
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
    --max_points 4096 \
    --sample_exponent 0.3
