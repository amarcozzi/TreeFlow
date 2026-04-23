#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name="tf-anim"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=log_animate_time_evolution.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python animate_time_evolution.py \
    --experiment_dir experiments/finetune-8-512-16384 \
    --data_path ./data/preprocessed-16384 \
    --tree_id 6069 \
    --cfg_scale 3.0 \
    --solver_method dopri5 \
    --seed 42 \
    --output_dir figures \
    --fps 25 \
    --duration_s 8.0 \
    --hold_frames 40 \
    --ease_power 3.0
