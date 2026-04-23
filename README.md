# TreeFlow

Flow matching for tree point cloud generation.

## Data Setup

### 1. Download the FOR-species20K Dataset

Download the dataset from the Zenodo repository:
- **Paper**: Puliti et al. "Benchmarking tree species classification from proximally sensed laser scanning data"
- **Data**: https://zenodo.org/records/13255198

Extract the data to `FOR-species20K/`:
```
FOR-species20K/
├── laz/
│   └── dev/          # LAZ point cloud files
├── tree_metadata_dev.csv
└── ...
```

### 2. Preprocess the Data

The preprocessing script converts LAZ files to normalized Zarr format and assigns train/val/test splits.

```bash
# Full dataset (all points)
python preprocess_laz.py --output_path ./data/preprocessed-full

# Filtered dataset (minimum 4096 points per tree)
python preprocess_laz.py --output_path ./data/preprocessed-4096 --min_points 4096

# Filtered dataset (minimum 16384 points per tree)
python preprocess_laz.py --output_path ./data/preprocessed-16384 --min_points 16384
```

This creates:
```
data/preprocessed-full/
├── 00001.zarr        # Normalized point clouds
├── 00002.zarr
├── ...
├── metadata.csv      # Metadata with train/val/test splits
└── preprocessing_summary.json
```

The normalization formula centers each point cloud and scales by height:
```
points_norm = (points - bbox_center) / height * 2.0
```

This produces points with Z in [-1, 1] range, matching the variance of Gaussian noise used in flow matching.

### 3. Train a Model

```bash
python train.py --data_path ./data/preprocessed-full --experiment_name my_experiment
```

Key arguments:
- `--data_path`: Path to preprocessed dataset (default: `data/preprocessed-full`)
- `--max_points`: Maximum points per sample (for memory/speed tradeoff)
- `--batch_size`: Training batch size
- `--num_epochs`: Number of training epochs

### 4. Generate Samples

```bash
python generate_samples.py \
    --experiment_name my_experiment \
    --data_path ./data/preprocessed-full \
    --num_samples_per_tree 5
```

## Directory Structure

```
TreeFlow/
├── FOR-species20K/       # Raw data from Zenodo (do not modify)
├── data/                 # Preprocessed datasets
│   ├── preprocessed-full/     # All trees
│   ├── preprocessed-4096/     # Trees with >= 4096 points
│   └── preprocessed-16384/    # Trees with >= 16384 points
├── experiments/          # Training outputs
├── preprocess_laz.py     # Data preprocessing
├── dataset.py            # PyTorch dataset
├── train.py              # Training script
├── generate_samples.py   # Sample generation
└── models/               # Model architectures
```
