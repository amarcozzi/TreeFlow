#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="train_flow_matching"
#SBATCH --cpus-per-task=48
#SBATCH --time=2-0
#SBATCH --output=log_train_flow_matching.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

python train.py \
    --voxel_size=0.1 \
    --batch_size=16 \
    --num_workers=36