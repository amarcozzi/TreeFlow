#!/bin/bash
#SBATCH --account=umontana_fire_modeling
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name="tf-gen-16384"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8
#SBATCH --output=log_generate_samples_16384_%a.out

module load cuda

source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate treeflow

TOTAL_TREES=1271
NUM_TASKS=8
TREES_PER_TASK=$(( (TOTAL_TREES + NUM_TASKS - 1) / NUM_TASKS ))
START_IDX=$(( SLURM_ARRAY_TASK_ID * TREES_PER_TASK ))
END_IDX=$(( START_IDX + TREES_PER_TASK ))
if [ $END_IDX -gt $TOTAL_TREES ]; then END_IDX=$TOTAL_TREES; fi

echo "Task $SLURM_ARRAY_TASK_ID: Processing trees [$START_IDX, $END_IDX)"

python generate_samples.py \
    --experiment_name finetune-8-512-16384 \
    --checkpoint epoch_3300.pt \
    --data_path ./data/preprocessed-16384 \
    --max_points 16384 \
    --num_samples_per_tree 16 \
    --cfg_scale "1.0 4.5" \
    --solver_method dopri5 \
    --batch_size 4 \
    --resume \
    --start_idx $START_IDX \
    --end_idx $END_IDX
