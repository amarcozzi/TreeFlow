#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:4
#SBATCH --job-name="pretrain_tf_8192"
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=2-0
#SBATCH --output=log_pretrain_transformer_8_256_8192.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

accelerate launch --num_processes=4 train.py \
    --output_dir experiments \
    --experiment_name "pretrain-8-256-8192" \
    --pretrained_weights ./experiments/pretrain-8-256-4096/checkpoints/epoch_1000.pt \
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
    --num_epochs 1000 \
    --lr 5e-5 \
    --lr_scheduler constant \
    --grad_clip_norm 0.5 \
    --mixed_precision bf16 \
    --cfg_dropout_prob 0.15 \
    --max_points 8192 \
    --sample_exponent 0.3