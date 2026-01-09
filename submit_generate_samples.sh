#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="treeflow-gen"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=log_generate_samples_%j.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

# ============================================
# Configuration - Edit these variables
# ============================================

# Experiment to generate from
EXPERIMENT_NAME="transformer-8-256-4096"
CHECKPOINT="epoch_15000.pt"  # "epoch_100.pt", etc.

# Data settings (should match training)
DATA_PATH="FOR-species20K"
CSV_PATH="FOR-species20K/tree_metadata_dev.csv"
PREPROCESSED_VERSION="raw"
MAX_POINTS=4096

# Generation settings
NUM_SAMPLES=2           # Number of samples per tree
CFG_SCALE="2.0 5.0"          # Single value (e.g., "3.0") or range (e.g., "2.0 5.0")
SOLVER_METHOD="dopri5"   # dopri5, euler, or midpoint

# Output
OUTPUT_DIR="generated_samples"
OUTPUT_FORMAT="laz"      # npy or laz

# ============================================
# Run generation
# ============================================

python generate_samples.py \
    --experiment_name ${EXPERIMENT_NAME} \
    --checkpoint ${CHECKPOINT} \
    --experiments_dir experiments \
    --data_path ${DATA_PATH} \
    --csv_path ${CSV_PATH} \
    --preprocessed_version ${PREPROCESSED_VERSION} \
    --max_points ${MAX_POINTS} \
    --num_samples_per_tree ${NUM_SAMPLES} \
    --cfg_scale "${CFG_SCALE}" \
    --solver_method ${SOLVER_METHOD} \
    --output_dir ${OUTPUT_DIR} \
    --output_format ${OUTPUT_FORMAT} \
