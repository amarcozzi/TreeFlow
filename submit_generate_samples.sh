#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name="tf-gen-1"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=log_generate_samples_0.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python generate_samples.py \
    --experiment_name transformer-8-512-4096 \
    --checkpoint epoch_5000.pt \
    --data_path ./data/preprocessed-4096 \
    --max_points 4096 \
    --num_samples_per_tree 32 \
    --cfg_scale "1.0 4.5" \
    --solver_method dopri5 \
    --resume \
    --start_idx 0 \
    --end_idx 200 \

# To resume an interrupted run, add: --resume
