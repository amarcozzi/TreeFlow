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

OUTPUT_DIR="generated_samples/transformer-8-256-16384"

python generate_samples.py \
    --experiment_name transformer-8-256-16384 \
    --checkpoint epoch_3000.pt \
    --data_path FOR-species20K \
    --csv_path FOR-species20K/tree_metadata_dev.csv \
    --preprocessed_version raw \
    --max_points 16384 \
    --num_samples_per_tree 16 \
    --cfg_scale "1.0 4" \
    --solver_method dopri5 \
    --output_dir ${OUTPUT_DIR} \
    --start_idx 0 \
    --end_idx 200 \
    --resume

# To resume an interrupted run, use: --resume ${OUTPUT_DIR}

# python postprocess_samples.py ${OUTPUT_DIR}
