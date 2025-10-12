#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="fm_train"
#SBATCH --cpus-per-task=36
#SBATCH --time=2-0
#SBATCH --output=log_train.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

srun python train.py \
    --data_path ./FOR-species20K \
    --preprocessed_version "voxel_0.1m" \
    --batch_size 8 \
    --batch_mode accumulate \
    --num_workers 24 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --time_embed_dim 256 \
    --visualize_every 10 \
    --save_every 20 \
