source /project/umontana_fire_modeling/anthony.marcozzi/miniforge3/etc/profile.d/conda.sh
conda activate canopy-flow

srun -A $ACCOUNT python preprocess_laz_to_npy.py \
    --data_path ./FOR-species20K \
    --output_path ./FOR-species20K/npy \
    --num_workers 40 \
    --verify