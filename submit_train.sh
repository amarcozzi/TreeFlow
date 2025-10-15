#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="fm_train"
#SBATCH --cpus-per-task=36
#SBATCH --time=2-0
#SBATCH --output=log_train_transformer.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

python train.py \
    --data_path FOR-species20K \
    --preprocessed_version voxel_0.2m \
    --model_dim 128 \
    --num_heads 8 \
    --num_layers 8 \
    --dropout 0.1 \
    --batch_size 8 \
    --visualize_every 10 \
    --batch_mode sample_to_min \
    --rotation_augment \
    --num_workers 24 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --ode_method dopri5 \
    use_amp \
    use_flash_attention