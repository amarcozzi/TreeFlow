#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="tf_12_256"
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=2-0
#SBATCH --output=log_train_transformer_12_256_raw.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python train.py \
    --output_dir experiments \
    --experiment_name "transformer-12-256-raw" \
    --data_path FOR-species20K \
    --csv_path FOR-species20K/tree_metadata_dev.csv \
    --preprocessed_version raw \
    --model_dim 256 \
    --num_heads 8 \
    --num_layers 12 \
    --dropout 0.1 \
    --batch_size 8 \
    --visualize_every 10 \
    --save_every 100 \
    --sample_exponent 0.3 \
    --rotation_augment \
    --shuffle_augment \
    --num_workers 24 \
    --num_epochs 2500 \
    --lr 1e-4 \
    --use_amp \
    --compile \
    --cfg_dropout_prob 0.1 \
    --max_points 8192
