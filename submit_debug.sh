#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="fm_debug"
#SBATCH --cpus-per-task=36
#SBATCH --time=2-0
#SBATCH --output=log_debug.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

srun python debug_profiling.py