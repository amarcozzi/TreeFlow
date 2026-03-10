#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=atlas
#SBATCH --job-name="fig-qual"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=log_figure_qualitative.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python figures.py \
    --figures qualitative \
    --experiment_dir experiments/finetune-8-512-16384 \
    --data_path data/preprocessed-16384 \
    --n_rows 4 \
    --n_generated 4 \
    --seed 42
