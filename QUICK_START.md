# Quick Start Guide

Get up and running with BMA MIL Classifier in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- Your BMA image dataset and labels CSV

## Installation

### 1. Clone or Download

```bash
cd /path/to/pile_level_classification_windsurf
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Setup Data

### 1. Organize Your Data

```bash
# Create data directory
mkdir -p data/images

# Move your CSV file
mv /path/to/BWM_label_data.csv data/

# Move your images
mv /path/to/your/images/* data/images/
```

### 2. Verify Data Structure

Your directory should look like:
```
data/
├── BWM_label_data.csv
└── images/
    ├── image1.JPG
    ├── image2.JPG
    └── ...
```

### 3. Check CSV Format

Your CSV should have columns:
- `pile`: Pile identifier
- `image_path`: Relative path to image (just filename if in data/images/)
- `BMA_label`: Class label (1-4)

Example:
```csv
pile,image_path,BMA_label
Pile_001,image001.JPG,1
Pile_001,image002.JPG,1
Pile_002,image003.JPG,2
```

## Configuration

Edit `configs/config.py` if needed:

```python
# Data paths (update if different)
DATA_PATH = 'data/BWM_label_data.csv'
IMAGE_DIR = 'data/images'

# Training settings (adjust based on your GPU)
BATCH_SIZE = 4  # Reduce if out of memory
NUM_EPOCHS = 50

# Augmentation (recommended settings)
ENABLE_GEOMETRIC_AUG = True
ENABLE_COLOR_AUG = False  # Disable for medical images
ENABLE_NOISE_AUG = False  # Disable for medical images
```

## Training

### Run Training

```bash
python scripts/train.py
```

### What Happens During Training

1. **Data Loading**: Loads CSV and splits into train/val/test
2. **Feature Extraction**: Extracts ViT-R50 features from patches
3. **Training**: Trains MIL model with early stopping
4. **Evaluation**: Tests on held-out test set
5. **Results**: Saves model, plots, and metrics

### Monitor Progress

Training will display:
```
Epoch 1/50, Batch 0, Loss: 1.3456
Epoch 1/50, Train Loss: 1.2345, Val Acc: 0.6500, Val F1: 0.6234
New best model saved with validation accuracy: 0.6500
```

### Expected Training Time

- **CPU**: ~2-4 hours (depending on dataset size)
- **GPU**: ~20-40 minutes

## Results

After training, check:

### 1. Model Checkpoint
```
models/best_bma_mil_model.pth
```

### 2. Training Plots
```
results/training_history.png
```
Shows:
- Training loss curve
- Validation accuracy curve
- Confusion matrix

### 3. Logs
```
logs/bma_mil_standard_YYYYMMDD_HHMMSS.log
logs/results_standard_YYYYMMDD_HHMMSS.txt
```

### 4. Console Output
```
Test Set Results:
==============================================================
Overall Accuracy: 0.8500
Weighted F1 Score: 0.8423

Per-Class F1 Scores:
  BMA Class 1: 0.8571
  BMA Class 2: 0.8333
  BMA Class 3: 0.8462
  BMA Class 4: 0.8500

Confusion Matrix:
==============================================================
Predicted →
True ↓    BMA1  BMA2  BMA3  BMA4
BMA 1:     12     1     0     0
BMA 2:      1    10     1     0
BMA 3:      0     1    11     1
BMA 4:      0     0     1    11
==============================================================
```

## Testing

### Run Tests (Optional)

```bash
# Unit tests
python tests/test_unit.py

# End-to-end tests
python tests/test_end_to_end.py

# Augmentation visualization
python tests/test_augmentation.py
```

## Common Issues

### Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:** Reduce batch size in `configs/config.py`:
```python
BATCH_SIZE = 2  # or even 1
```

### Data Not Found

**Error:** `FileNotFoundError: data/BWM_label_data.csv`

**Solution:** 
1. Verify file exists: `ls data/BWM_label_data.csv`
2. Check path in `configs/config.py`

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Run from project root:
```bash
cd /path/to/pile_level_classification_windsurf
python scripts/train.py
```

### Slow Training

**Issue:** Training is very slow

**Solutions:**
1. Use GPU if available (check `Config.DEVICE`)
2. Reduce `MAX_IMAGES_PER_PILE` in config
3. Use fewer epochs for initial testing

## Next Steps

### 1. Experiment with Configuration

Try different settings in `configs/config.py`:
- Adjust learning rate
- Enable/disable augmentations
- Change model architecture dimensions

### 2. Cross-Validation

For more robust evaluation:
```python
# In configs/config.py
USE_CROSS_VALIDATION = True
NUM_FOLDS = 3
```

Then run:
```bash
python scripts/train.py
```

### 3. Hyperparameter Tuning

Experiment with:
- Learning rate: `1e-3`, `1e-4`, `1e-5`
- Batch size: `2`, `4`, `8`
- Hidden dimensions: `256`, `512`, `1024`

### 4. Custom Augmentation

Adjust augmentation in `configs/config.py`:
```python
ROTATION_RANGE = 30  # More rotation
ZOOM_RANGE = (0.8, 1.2)  # More zoom variation
```

## Tips for Best Results

### 1. Data Quality
- Ensure consistent image quality
- Verify labels are correct
- Check for class imbalance

### 2. Training
- Use early stopping (enabled by default)
- Monitor validation accuracy
- Save best model checkpoint

### 3. Evaluation
- Check confusion matrix for systematic errors
- Review per-class F1 scores
- Analyze attention weights (if needed)

### 4. Augmentation
- Start with geometric augmentation only
- Add color/noise if needed
- Validate augmentation improves performance

## Help & Support

### Documentation
- `README.md` - Full documentation
- `MIGRATION_GUIDE.md` - Import and usage changes
- `REORGANIZATION_SUMMARY.md` - Code structure details

### Troubleshooting
1. Check configuration in `configs/config.py`
2. Verify data paths and format
3. Review logs in `logs/` directory
4. Run tests to verify setup

## Summary

You're now ready to train! The basic workflow is:

1. **Setup**: Install dependencies, organize data
2. **Configure**: Edit `configs/config.py` if needed
3. **Train**: Run `python scripts/train.py`
4. **Evaluate**: Check results in `results/` and `logs/`
5. **Iterate**: Adjust configuration and retrain

Happy training! 🚀
