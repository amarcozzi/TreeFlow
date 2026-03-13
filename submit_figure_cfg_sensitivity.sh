#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=atlas
#SBATCH --job-name="fig-cfg"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=log_figure_cfg_sensitivity.out

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

python figures.py \
    --figures cfg_sensitivity \
    --experiment_dir experiments/finetune-8-512-16384
