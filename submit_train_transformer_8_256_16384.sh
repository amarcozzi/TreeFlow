#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="train_tf_16384"
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --time=2-0
#SBATCH --output=log_train_transformer_8_256_16384.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python train.py \
    --output_dir experiments \
    --experiment_name "transformer-8-256-16384-zarr" \
    --data_path FOR-species20K \
    --csv_path FOR-species20K/tree_metadata_dev.csv \
    --model_dim 256 \
    --num_heads 8 \
    --num_layers 8 \
    --dropout 0.1 \
    --batch_size 32 \
    --visualize_every 10 \
    --save_every 100 \
    --rotation_augment \
    --shuffle_augment \
    --num_workers 16 \
    --num_epochs 10000 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_scheduler cosine \
    --grad_clip_norm 1.0 \
    --use_amp \
    --compile \
    --cfg_dropout_prob 0.15 \
    --max_points 16384