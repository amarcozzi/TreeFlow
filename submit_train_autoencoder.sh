#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="train_autoencoder_parallel_slots"
#SBATCH --cpus-per-task=12
#SBATCH --time=2-0
#SBATCH --output=log_train_autoencoder_parallel_slots.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

srun python train_autoencoder.py