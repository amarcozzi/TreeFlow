#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name="finetune_tf_512_16k"
#SBATCH --cpus-per-task=20
#SBATCH --mem=512G
#SBATCH --time=2-0
#SBATCH --output=log_finetune_transformer_8_512_16384.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

accelerate launch --num_processes=4 train.py \
    --output_dir experiments \
    --model_type transformer \
    --experiment_name "finetune-8-512-16384" \
    --pretrained_weights ./experiments/pretrain-8-512-8192/checkpoints/epoch_1000.pt \
    --data_path ./data/preprocessed-16384 \
    --model_dim 512 \
    --num_heads 8 \
    --num_layers 8 \
    --dropout 0.1 \
    --batch_size 8 \
    --visualize_every 10 \
    --save_every 100 \
    --rotation_augment \
    --shuffle_augment \
    --num_workers 4 \
    --num_epochs 2000 \
    --resume \
    --lr 5e-5 \
    --min_lr 1e-6 \
    --lr_scheduler cosine \
    --grad_clip_norm 1.0 \
    --mixed_precision bf16 \
    --cfg_dropout_prob 0.15 \
    --max_points 16384 \
    --sample_exponent 0.3