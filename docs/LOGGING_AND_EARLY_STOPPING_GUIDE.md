# Logging and Early Stopping Guide

## Overview

The BMA MIL Classifier now includes **comprehensive logging** and **early stopping** features to improve training efficiency and result tracking.

## Features

### ✅ Logging System

1. **Dual Output**: Logs to both console and file simultaneously
2. **Timestamped Log Files**: Each run creates a unique log file with timestamp
3. **Structured Logging**: INFO, DEBUG, WARNING, ERROR levels
4. **Training Metrics**: All epochs, losses, accuracies logged
5. **Results Summary**: Automatic results file generation

### ✅ Early Stopping

1. **Automatic Training Termination**: Stops when validation accuracy plateaus
2. **Configurable Patience**: Set how many epochs to wait
3. **Minimum Delta**: Define minimum improvement threshold
4. **Best Model Tracking**: Saves best model before stopping

## Configuration

### In `config.py`:

```python
# Early stopping
USE_EARLY_STOPPING = True  # Enable early stopping
EARLY_STOPPING_PATIENCE = 10  # Number of epochs to wait for improvement
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to qualify as improvement

# Logging
ENABLE_LOGGING = True  # Enable logging to file
LOG_LEVEL = 'INFO'  # Logging level: DEBUG, INFO, WARNING, ERROR
LOG_DIR = 'logs'  # Directory for log files
```

## Logging Features

### Log Files Generated

1. **Training Log**: `logs/bma_mil_standard_YYYYMMDD_HHMMSS.log`
   - All training progress
   - Epoch-by-epoch metrics
   - Model save events
   - Early stopping triggers

2. **Results Summary**: `logs/results_standard_YYYYMMDD_HHMMSS.txt`
   - Final accuracy and F1 scores
   - Confusion matrix
   - Configuration used
   - Per-class metrics

### Log File Structure

```
2025-09-30 15:03:30 - INFO - Logging initialized. Log file: logs/bma_mil_standard_20250930_150330.log
2025-09-30 15:03:30 - INFO - Mode: standard
2025-09-30 15:03:30 - INFO - Device: cuda
2025-09-30 15:03:30 - INFO - Configuration: Epochs=50, Batch Size=4, LR=0.0001
2025-09-30 15:03:30 - INFO - Augmentation: Geometric=True, Color=False, Noise=False
2025-09-30 15:03:35 - INFO - Using weighted CrossEntropyLoss for class imbalance
2025-09-30 15:03:35 - INFO - Early stopping enabled: patience=10, min_delta=0.001
2025-09-30 15:05:12 - INFO - Epoch 1/50, Train Loss: 1.3245, Val Acc: 0.6364, Val F1: 0.6123
2025-09-30 15:06:45 - INFO - New best model saved with validation accuracy: 0.6364
...
2025-09-30 15:45:23 - INFO - Epoch 25/50, Train Loss: 0.2156, Val Acc: 0.8485, Val F1: 0.8456
2025-09-30 15:45:23 - INFO - New best model saved with validation accuracy: 0.8485
2025-09-30 15:47:01 - INFO - EarlyStopping counter: 1/10 (best: 0.8485 at epoch 25)
...
2025-09-30 16:05:12 - INFO - Early stopping triggered at epoch 35. Best accuracy: 0.8485 at epoch 25
2025-09-30 16:05:15 - INFO - Test Results: Accuracy = 0.8500, F1 = 0.8450
2025-09-30 16:05:15 - INFO - Results saved to: logs/results_standard_20250930_160515.txt
```

### Results Summary File

```
============================================================
BMA MIL Classifier - Results Summary
Mode: standard
Timestamp: 2025-09-30 16:05:15
============================================================

Configuration:
  Epochs: 50
  Batch Size: 4
  Learning Rate: 0.0001
  Early Stopping: True
  Patience: 10
  Weighted Loss: True
  Device: cuda

Test Set Results:
============================================================
Overall Accuracy: 0.8500
Weighted F1 Score: 0.8450

Per-Class F1 Scores:
  BMA Class 1: 0.8200
  BMA Class 2: 0.8500
  BMA Class 3: 0.8700
  BMA Class 4: 0.8400

Confusion Matrix:
Predicted →
True ↓    BMA1  BMA2  BMA3  BMA4
BMA 1:    10     1     0     0  
BMA 2:     1    12     1     0  
BMA 3:     0     1    11     0  
BMA 4:     0     0     1     9  
```

## Early Stopping

### How It Works

1. **Monitors Validation Accuracy**: Tracks improvement after each epoch
2. **Patience Counter**: Increments when no improvement detected
3. **Minimum Delta**: Requires improvement > `min_delta` to reset counter
4. **Triggers Stop**: When counter reaches `patience`, training stops
5. **Best Model Saved**: Always saves the best model before stopping

### Example Output

```
Epoch 25/50, Train Loss: 0.2156, Val Acc: 0.8485, Val F1: 0.8456
New best model saved with validation accuracy: 0.8485

Epoch 26/50, Train Loss: 0.2134, Val Acc: 0.8470, Val F1: 0.8445
EarlyStopping counter: 1/10 (best: 0.8485 at epoch 25)

Epoch 27/50, Train Loss: 0.2112, Val Acc: 0.8455, Val F1: 0.8430
EarlyStopping counter: 2/10 (best: 0.8485 at epoch 25)

...

Epoch 35/50, Train Loss: 0.2001, Val Acc: 0.8440, Val F1: 0.8420
EarlyStopping counter: 10/10 (best: 0.8485 at epoch 25)
Early stopping triggered at epoch 35. Best accuracy: 0.8485 at epoch 25
```

### Benefits

1. **Prevents Overfitting**: Stops before model starts overfitting
2. **Saves Time**: No need to run all epochs if converged
3. **Automatic**: No manual monitoring required
4. **Optimal Model**: Always uses best validation performance

## Usage Examples

### Example 1: Enable Both Features (Recommended)

```python
# In config.py
Config.USE_EARLY_STOPPING = True
Config.EARLY_STOPPING_PATIENCE = 10
Config.EARLY_STOPPING_MIN_DELTA = 0.001
Config.ENABLE_LOGGING = True
Config.LOG_LEVEL = 'INFO'
```

```bash
python bma_mil_classifier.py
```

### Example 2: Disable Early Stopping (Run All Epochs)

```python
# In config.py
Config.USE_EARLY_STOPPING = False
Config.ENABLE_LOGGING = True
```

### Example 3: Debug Mode (Verbose Logging)

```python
# In config.py
Config.ENABLE_LOGGING = True
Config.LOG_LEVEL = 'DEBUG'  # More detailed logs
```

### Example 4: Quick Testing (Aggressive Early Stopping)

```python
# In config.py
Config.USE_EARLY_STOPPING = True
Config.EARLY_STOPPING_PATIENCE = 3  # Stop after 3 epochs without improvement
Config.EARLY_STOPPING_MIN_DELTA = 0.005  # Require 0.5% improvement
```

## Configuration Parameters

### Early Stopping Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_EARLY_STOPPING` | `True` | Enable/disable early stopping |
| `EARLY_STOPPING_PATIENCE` | `10` | Epochs to wait for improvement |
| `EARLY_STOPPING_MIN_DELTA` | `0.001` | Minimum improvement (0.1%) |

**Recommendations:**
- **Small datasets**: `patience=5-10`, `min_delta=0.001`
- **Large datasets**: `patience=15-20`, `min_delta=0.0005`
- **Quick testing**: `patience=3`, `min_delta=0.01`

### Logging Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_LOGGING` | `True` | Enable/disable logging |
| `LOG_LEVEL` | `'INFO'` | Logging verbosity |
| `LOG_DIR` | `'logs'` | Directory for log files |

**Log Levels:**
- `DEBUG`: Very detailed (every batch)
- `INFO`: Standard (epoch summaries)
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

## Cross-Validation Mode

Early stopping and logging work seamlessly with cross-validation:

```python
Config.USE_CROSS_VALIDATION = True
Config.NUM_FOLDS = 3
Config.USE_EARLY_STOPPING = True
Config.ENABLE_LOGGING = True
```

**Log Files Generated:**
- `logs/bma_mil_cross_validation_YYYYMMDD_HHMMSS.log`
- `logs/results_cross_validation_YYYYMMDD_HHMMSS.txt`

**Features:**
- Each fold can stop early independently
- Separate best models saved per fold: `best_bma_mil_model_fold1.pth`, etc.
- Comprehensive summary across all folds

## Analyzing Results

### View Log Files

```bash
# View latest log
cat logs/bma_mil_standard_*.log | tail -100

# Search for specific metrics
grep "Val Acc" logs/bma_mil_standard_*.log

# Check early stopping events
grep "Early stopping" logs/bma_mil_standard_*.log
```

### View Results Summary

```bash
# View latest results
cat logs/results_standard_*.txt

# Compare multiple runs
ls -lt logs/results_*.txt | head -5
```

## Troubleshooting

### Issue: Early Stopping Too Aggressive

**Symptoms**: Training stops after just a few epochs

**Solutions:**
1. Increase patience: `EARLY_STOPPING_PATIENCE = 15`
2. Decrease min_delta: `EARLY_STOPPING_MIN_DELTA = 0.0005`
3. Check if learning rate is too high

### Issue: Early Stopping Never Triggers

**Symptoms**: Training runs all epochs, never stops early

**Solutions:**
1. Decrease patience: `EARLY_STOPPING_PATIENCE = 5`
2. Increase min_delta: `EARLY_STOPPING_MIN_DELTA = 0.005`
3. Model might be continuously improving (good!)

### Issue: Log Files Too Large

**Solutions:**
1. Change log level to INFO: `LOG_LEVEL = 'INFO'`
2. Reduce batch logging frequency in code
3. Periodically clean old logs

### Issue: Can't Find Log Files

**Check:**
1. Verify `ENABLE_LOGGING = True`
2. Check `LOG_DIR` path exists
3. Look in current directory for `logs/` folder

## Best Practices

### 1. Always Enable Logging for Important Runs

```python
Config.ENABLE_LOGGING = True
Config.LOG_LEVEL = 'INFO'
```

### 2. Use Early Stopping for Efficiency

```python
Config.USE_EARLY_STOPPING = True
Config.EARLY_STOPPING_PATIENCE = 10
```

### 3. Archive Important Results

```bash
# Create experiment directory
mkdir experiments/exp_001
cp logs/results_*.txt experiments/exp_001/
cp logs/bma_mil_*.log experiments/exp_001/
cp best_bma_mil_model.pth experiments/exp_001/
```

### 4. Compare Multiple Runs

Keep a spreadsheet or notebook tracking:
- Timestamp
- Configuration used
- Final accuracy
- Early stopping epoch
- Log file path

### 5. Monitor Training Progress

```bash
# Watch training in real-time
tail -f logs/bma_mil_standard_*.log
```

## Integration with Existing Features

### Works With:
- ✅ Cross-validation mode
- ✅ Data augmentation
- ✅ Class weighting
- ✅ Confusion matrix visualization
- ✅ Per-class F1 scores

### Backward Compatible:
- ✅ Can disable both features (legacy mode)
- ✅ Existing code continues to work
- ✅ No breaking changes

## Summary

**Logging Benefits:**
- Track all experiments automatically
- Compare different configurations
- Debug training issues
- Share results easily

**Early Stopping Benefits:**
- Save training time (often 30-50%)
- Prevent overfitting
- Automatic optimal model selection
- No manual monitoring needed

**Combined Benefits:**
- Complete training history preserved
- Know exactly when and why training stopped
- Reproducible experiments
- Professional ML workflow
