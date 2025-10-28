#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="tf_12_512"
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=2-0
#SBATCH --output=log_train_transformer_12_512.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

python train.py \
    --output_dir experiments \
    --experiment_name "transformer-12-512" \
    --data_path FOR-species20K \
    --preprocessed_version voxel_0.2m \
    --model_dim 512 \
    --num_heads 16 \
    --num_layers 12 \
    --dropout 0.1 \
    --batch_size 8 \
    --visualize_every 5 \
    --batch_mode accumulate \
    --sample_exponent 0.3 \
    --rotation_augment \
    --shuffle_augment \
    --num_workers 24 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --ode_method dopri5 \
    --use_amp \
    --use_flash_attention \
    --min_visualization_points 1000 \
    --max_visualization_points 8000 \
    --resume_from "experiments/transformer-12-512/checkpoints/checkpoint_epoch_500.pt" \
    --load_weights_only