#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="tf_48_256"
#SBATCH --cpus-per-task=36
#SBATCH --time=2-0
#SBATCH --output=log_train_transformer_48_256.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

python train.py \
    --output_dir experiments \
    --experiment_name "transformer-48-256" \
    --data_path FOR-species20K \
    --preprocessed_version voxel_0.1m \
    --model_dim 256 \
    --num_heads 8 \
    --num_layers 48 \
    --dropout 0.1 \
    --batch_size 8 \
    --visualize_every 5 \
    --batch_mode sample_to_min \
    --rotation_augment \
    --num_workers 24 \
    --num_epochs 2000 \
    --lr 1e-4 \
    --ode_method dopri5 \
    --use_amp \
    --use_flash_attention \
    --min_visualization_points 2500 \
    --max_visualization_points 50000