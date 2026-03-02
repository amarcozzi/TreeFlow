#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name="fig-qual"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=log_figure_qualitative.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python figures.py \
    --figures qualitative \
    --experiment_dir experiments/finetune-8-512-16384 \
    --data_path data/preprocessed-16384 \
    --cfg_scale "1.5,5.0" \
    --n_generated 5 \
    --seed 42
