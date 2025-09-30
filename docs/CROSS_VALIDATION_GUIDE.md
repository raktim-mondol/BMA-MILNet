# Cross-Validation Guide for BMA MIL Classifier

## Overview

The BMA MIL Classifier now supports **3-fold stratified cross-validation** as an alternative to the standard train/val/test split. This provides more robust performance estimates by training and evaluating the model on different data splits.

## Features

### ✅ What's Included

1. **Stratified K-Fold Cross-Validation**: Ensures balanced class distribution across all folds
2. **Overall Accuracy**: Computed for each fold and averaged across all folds
3. **Per-Class F1 Scores**: Calculated for each BMA class (1-4) in each fold
4. **Comprehensive Metrics**: Mean ± Standard Deviation for all metrics
5. **Visualization**: 6-panel plot showing:
   - Training loss per fold
   - Validation accuracy per fold
   - Final accuracy per fold
   - Per-class F1 scores per fold
   - Mean F1 scores with error bars
   - Average confusion matrix

## How to Use

### Method 1: Enable in Config

Edit `config.py`:

```python
# Cross-validation
USE_CROSS_VALIDATION = True  # Set to True to use k-fold cross-validation
NUM_FOLDS = 3  # Number of folds for cross-validation
```

Then run:
```bash
python bma_mil_classifier.py
```

### Method 2: Use the Dedicated Script

Simply run:
```bash
python run_cross_validation.py
```

This script automatically enables cross-validation mode.

### Method 3: Programmatic Usage

```python
from config import Config
from bma_mil_classifier import cross_validate_model
import pandas as pd

# Load data
df = pd.read_csv('BWM_label_data.csv')
df = df[df['BMA_label'] != 'BMA_label']
df['BMA_label'] = df['BMA_label'].astype(int)

# Run cross-validation
fold_results, mean_accuracy, mean_f1_per_class = cross_validate_model(
    df=df,
    image_dir='.',
    num_folds=3,
    num_epochs=50,
    learning_rate=1e-4
)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
for i, f1 in enumerate(mean_f1_per_class):
    print(f"BMA Class {i+1} F1: {f1:.4f}")
```

## Output Metrics

### Per-Fold Metrics

For each fold, you'll see:

```
============================================================
Fold 1 Results:
============================================================
Overall Accuracy: 0.8500
Weighted F1 Score: 0.8450

Per-Class F1 Scores:
  BMA Class 1: 0.8200
  BMA Class 2: 0.8500
  BMA Class 3: 0.8700
  BMA Class 4: 0.8400

Confusion Matrix:
[[10  1  0  0]
 [ 1 12  1  0]
 [ 0  1 11  0]
 [ 0  0  1  9]]
```

### Summary Metrics

After all folds complete:

```
============================================================
Cross-Validation Summary (3 Folds)
============================================================

Overall Accuracy: 0.8467 ± 0.0125
Accuracy per fold: ['0.8500', '0.8400', '0.8500']

Per-Class F1 Scores (Mean ± Std):
  BMA Class 1: 0.8233 ± 0.0153
  BMA Class 2: 0.8467 ± 0.0115
  BMA Class 3: 0.8633 ± 0.0115
  BMA Class 4: 0.8367 ± 0.0115
```

## Visualization

The cross-validation generates a comprehensive plot saved as `cross_validation_results.png` with 6 subplots:

1. **Training Loss per Fold**: Shows convergence behavior for each fold
2. **Validation Accuracy per Fold**: Tracks validation performance during training
3. **Final Accuracy per Fold**: Bar chart comparing final accuracy across folds
4. **Per-Class F1 Scores**: Grouped bar chart showing F1 for each class in each fold
5. **Mean F1 Score per Class**: Bar chart with error bars showing mean ± std
6. **Average Confusion Matrix**: Heatmap of average predictions across all folds

## Configuration Options

In `config.py`:

```python
# Cross-validation settings
USE_CROSS_VALIDATION = False  # Enable/disable CV mode
NUM_FOLDS = 3                 # Number of folds (typically 3, 5, or 10)

# Training parameters (apply to both modes)
NUM_EPOCHS = 50               # Epochs per fold in CV mode
BATCH_SIZE = 4                # Batch size
LEARNING_RATE = 1e-4          # Learning rate
USE_WEIGHTED_LOSS = True      # Handle class imbalance

# Data augmentation (apply to both modes)
ENABLE_GEOMETRIC_AUG = True   # Geometric augmentation
ENABLE_COLOR_AUG = False      # Color augmentation
ENABLE_NOISE_AUG = False      # Noise/blur augmentation
```

## Comparison: CV vs Train/Val/Test

| Feature | Cross-Validation | Train/Val/Test |
|---------|------------------|----------------|
| **Data Usage** | All data used for training & validation | 70% train, 15% val, 15% test |
| **Model Evaluation** | Average across 3 folds | Single test set |
| **Robustness** | More robust estimates | May depend on split |
| **Training Time** | 3× longer (3 models) | 1× (single model) |
| **Best For** | Small datasets, model selection | Large datasets, final deployment |
| **Output** | Mean ± Std metrics | Single metrics |

## Best Practices

### When to Use Cross-Validation

✅ **Use CV when:**
- Dataset is small (< 100 piles)
- You want robust performance estimates
- Comparing different model architectures or hyperparameters
- Publishing research results

❌ **Use Train/Val/Test when:**
- Dataset is large (> 200 piles)
- You need a final model for deployment
- Training time is a constraint
- You have a separate held-out test set

### Interpreting Results

1. **High Standard Deviation**: Indicates model performance varies significantly across folds
   - May suggest overfitting or insufficient data
   - Consider increasing regularization or gathering more data

2. **Low Standard Deviation**: Indicates consistent performance
   - Good sign of model stability
   - Results are more reliable

3. **Per-Class F1 Scores**: Identify which classes are harder to classify
   - Low F1 for a class → Need more data or better features for that class
   - Imbalanced F1 scores → Consider class-specific augmentation

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or number of epochs
```python
Config.BATCH_SIZE = 2
Config.NUM_EPOCHS = 25
```

### Issue: Training Takes Too Long

**Solution**: Reduce epochs for testing
```python
Config.NUM_EPOCHS = 10  # Quick test
```

### Issue: Poor Performance on Some Folds

**Possible Causes**:
- Class imbalance in folds (check class distribution output)
- Insufficient data for stratification
- Random initialization differences

**Solutions**:
- Ensure `USE_WEIGHTED_LOSS = True`
- Increase `NUM_FOLDS` (e.g., 5 or 10)
- Set `RANDOM_STATE` for reproducibility

## Example Output

Here's what a complete cross-validation run looks like:

```
============================================================
BMA MIL Classifier - Training Pipeline
============================================================
Mode: Cross-Validation
Device: cuda
============================================================

============================================================
Starting 3-Fold Cross-Validation
============================================================

============================================================
Fold 1/3
============================================================
Training piles: 67
Validation piles: 33
Train - BMA classes: {1: 17, 2: 17, 3: 17, 4: 16}
Val   - BMA classes: {1: 8, 2: 8, 3: 8, 4: 9}

Training fold 1...
Epoch 1/50, Train Loss: 1.3245, Val Acc: 0.6364
...
Epoch 50/50, Train Loss: 0.2156, Val Acc: 0.8485

Evaluating fold 1...

============================================================
Fold 1 Results:
============================================================
Overall Accuracy: 0.8485
Weighted F1 Score: 0.8456

Per-Class F1 Scores:
  BMA Class 1: 0.8421
  BMA Class 2: 0.8500
  BMA Class 3: 0.8571
  BMA Class 4: 0.8333

[... Folds 2 and 3 ...]

============================================================
Cross-Validation Summary (3 Folds)
============================================================

Overall Accuracy: 0.8467 ± 0.0125
Accuracy per fold: ['0.8485', '0.8400', '0.8515']

Per-Class F1 Scores (Mean ± Std):
  BMA Class 1: 0.8421 ± 0.0153
  BMA Class 2: 0.8467 ± 0.0115
  BMA Class 3: 0.8571 ± 0.0115
  BMA Class 4: 0.8333 ± 0.0115

Cross-validation plots saved to 'cross_validation_results.png'

============================================================
Cross-Validation Completed!
============================================================
Mean Accuracy: 0.8467

Mean F1 Scores per Class:
  BMA Class 1: 0.8421
  BMA Class 2: 0.8467
  BMA Class 3: 0.8571
  BMA Class 4: 0.8333
```

## Additional Notes

- **Stratification**: The cross-validation uses stratified splits to maintain class balance
- **Feature Extractor**: Shared across all folds for efficiency
- **Augmentation**: Training augmentation applied to training folds only
- **Class Weights**: Computed separately for each fold if `USE_WEIGHTED_LOSS = True`
- **Model Saving**: Each fold trains a separate model (not saved by default in CV mode)

## References

- Scikit-learn StratifiedKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
- Cross-validation best practices: https://scikit-learn.org/stable/modules/cross_validation.html
