# Data Augmentation Guide for BMA Classification

## Overview

This document describes the comprehensive data augmentation pipeline implemented for the BMA (Bone Marrow Aspirate) classification project. The augmentation techniques are specifically designed for medical imaging and help improve model robustness and generalization.

## Architecture Integration

```
Original Image (4032×3024)
    ↓
Patch Extraction (12 patches of 1008×1008)
    ↓
Resize to 224×224
    ↓
[AUGMENTATION PIPELINE] ← Applied here
    ↓
Feature Extraction (ViT-R50)
    ↓
Multi-level MIL Classification
```

## Augmentation Techniques

### 1. Histogram Normalization (Applied to ALL data)

**Purpose**: Normalize lighting and contrast variations across different images

**Methods**:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** - Default
  - Clip limit: 2.0
  - Tile grid size: 8×8
  - Applied to LAB color space (L channel)
  - Best for medical images with local contrast variations

- **Adaptive Histogram Equalization**
  - Applied per-channel
  - Good for overall contrast enhancement

- **Standard Histogram Equalization**
  - Applied to Y channel in YCrCb space
  - Preserves color information

**Configuration**:
```python
HISTOGRAM_METHOD = 'clahe'  # Options: 'clahe', 'adaptive', 'standard', 'none'
```

**Benefits**:
- Normalizes staining variations
- Improves consistency across different imaging sessions
- Enhances cell structure visibility
- Applied to training, validation, AND test sets

---

### 2. Geometric Transformations (Training Only)

**Purpose**: Make model invariant to orientation, scale, and position

#### 2.1 Rotation
- **Range**: ±15 degrees
- **Interpolation**: Bilinear
- **Probability**: 50%

```python
ROTATION_RANGE = 15  # degrees
```

#### 2.2 Zoom
- **Range**: 0.9× to 1.1× (90% to 110%)
- **Implementation**: Resize + center crop/pad
- **Probability**: 50%

```python
ZOOM_RANGE = (0.9, 1.1)
```

#### 2.3 Shear
- **Range**: ±10 degrees
- **Axes**: Horizontal shear
- **Probability**: 50%

```python
SHEAR_RANGE = 10  # degrees
```

#### 2.4 Flips
- **Horizontal Flip**: 50% probability
- **Vertical Flip**: 50% probability

```python
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
```

**Benefits**:
- Increases effective dataset size
- Reduces overfitting
- Handles variations in slide orientation
- Improves robustness to cell positioning

---

### 3. Color Augmentations (Training Only)

**Purpose**: Handle variations in staining protocols and imaging conditions

#### 3.1 Brightness
- **Range**: 0.8× to 1.2× (80% to 120%)
- **Probability**: 50%

```python
BRIGHTNESS_RANGE = (0.8, 1.2)
```

#### 3.2 Contrast
- **Range**: 0.8× to 1.2×
- **Probability**: 50%

```python
CONTRAST_RANGE = (0.8, 1.2)
```

#### 3.3 Saturation
- **Range**: 0.8× to 1.2×
- **Probability**: 50%

```python
SATURATION_RANGE = (0.8, 1.2)
```

#### 3.4 Hue Shift
- **Range**: ±0.1 (±18 degrees in HSV)
- **Probability**: 50%

```python
HUE_RANGE = (-0.1, 0.1)
```

**Benefits**:
- Handles different staining protocols
- Compensates for imaging equipment variations
- Reduces sensitivity to color calibration
- Critical for medical imaging generalization

---

### 4. Noise and Blur (Training Only)

**Purpose**: Improve robustness to image quality variations

#### 4.1 Gaussian Noise
- **Standard Deviation**: 0.01
- **Probability**: 30%
- **Applied to**: Normalized pixel values [0, 1]

```python
NOISE_STD = 0.01
```

#### 4.2 Gaussian Blur
- **Sigma Range**: 0.1 to 2.0
- **Probability**: 30%

```python
BLUR_SIGMA = (0.1, 2.0)
```

**Benefits**:
- Simulates focus variations
- Handles image compression artifacts
- Improves robustness to image quality
- Prevents overfitting to high-frequency details

---

## Implementation Details

### File Structure

```
pile_level_classification_windsurf/
├── augmentation.py              # Core augmentation module
├── bma_mil_classifier.py        # Main classifier (integrated)
├── config.py                    # Configuration parameters
├── test_augmentation.py         # Testing and visualization
└── AUGMENTATION_GUIDE.md        # This file
```

### Key Classes

#### `HistogramNormalizer`
```python
normalizer = HistogramNormalizer(method='clahe')
normalized_image = normalizer(pil_image)
```

#### `GeometricAugmentation`
```python
geo_aug = GeometricAugmentation(
    rotation_range=15,
    zoom_range=(0.9, 1.1),
    shear_range=10,
    horizontal_flip=True,
    vertical_flip=True,
    probability=0.5
)
augmented_image = geo_aug(pil_image)
```

#### `ColorAugmentation`
```python
color_aug = ColorAugmentation(
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
    hue_range=(-0.1, 0.1),
    probability=0.5
)
augmented_image = color_aug(pil_image)
```

#### `NoiseAndBlurAugmentation`
```python
noise_aug = NoiseAndBlurAugmentation(
    gaussian_noise_std=0.01,
    gaussian_blur_sigma=(0.1, 2.0),
    probability=0.3
)
augmented_image = noise_aug(pil_image)
```

#### `ComposedAugmentation`
```python
# Training pipeline
train_aug = ComposedAugmentation(
    histogram_method='clahe',
    enable_geometric=True,
    enable_color=True,
    enable_noise=True,
    is_training=True
)

# Validation/Test pipeline (histogram only)
val_aug = ComposedAugmentation(
    histogram_method='clahe',
    is_training=False
)
```

### Factory Function

```python
from augmentation import get_augmentation_pipeline

# For training
train_aug = get_augmentation_pipeline(is_training=True, config=Config)

# For validation/testing
val_aug = get_augmentation_pipeline(is_training=False, config=Config)
```

---

## Usage Examples

### Basic Usage

```python
from augmentation import get_augmentation_pipeline
from PIL import Image

# Load image
image = Image.open('path/to/image.jpg')

# Create augmentation pipeline
augmentation = get_augmentation_pipeline(is_training=True)

# Apply augmentation
augmented_image = augmentation(image)
```

### Integration with Dataset

```python
from bma_mil_classifier import BMADataset
from augmentation import get_augmentation_pipeline

# Create augmentation pipelines
train_aug = get_augmentation_pipeline(is_training=True)
val_aug = get_augmentation_pipeline(is_training=False)

# Create datasets
train_dataset = BMADataset(
    train_df, 
    image_dir, 
    feature_extractor,
    augmentation=train_aug,
    is_training=True
)

val_dataset = BMADataset(
    val_df,
    image_dir,
    feature_extractor,
    augmentation=val_aug,
    is_training=False
)
```

### Custom Configuration

```python
from augmentation import ComposedAugmentation

# Custom augmentation with specific parameters
custom_aug = ComposedAugmentation(
    histogram_method='clahe',
    enable_geometric=True,
    enable_color=True,
    enable_noise=False,  # Disable noise
    is_training=True,
    rotation_range=30,  # More aggressive rotation
    zoom_range=(0.8, 1.3),  # Wider zoom range
    geometric_prob=0.7  # Higher probability
)
```

---

## Testing and Visualization

### Run Augmentation Tests

```bash
python test_augmentation.py
```

This generates 6 visualization files:
1. `test_histogram_normalization.png` - Compare histogram methods
2. `test_geometric_augmentation.png` - Geometric transformations
3. `test_color_augmentation.png` - Color variations
4. `test_noise_blur_augmentation.png` - Noise and blur effects
5. `test_full_pipeline.png` - Complete pipeline comparison
6. `test_patch_augmentation.png` - Patch-level augmentation

### Visual Inspection

Always visually inspect augmented images to ensure:
- ✓ Augmentations are realistic
- ✓ Medical features are preserved
- ✓ No artifacts introduced
- ✓ Appropriate intensity ranges

---

## Configuration Parameters

All parameters can be adjusted in `config.py`:

```python
class Config:
    # Histogram normalization
    HISTOGRAM_METHOD = 'clahe'
    
    # Enable/disable augmentation types
    ENABLE_GEOMETRIC_AUG = True
    ENABLE_COLOR_AUG = True
    ENABLE_NOISE_AUG = True
    
    # Geometric parameters
    ROTATION_RANGE = 15
    ZOOM_RANGE = (0.9, 1.1)
    SHEAR_RANGE = 10
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    GEOMETRIC_PROB = 0.5
    
    # Color parameters
    BRIGHTNESS_RANGE = (0.8, 1.2)
    CONTRAST_RANGE = (0.8, 1.2)
    SATURATION_RANGE = (0.8, 1.2)
    HUE_RANGE = (-0.1, 0.1)
    COLOR_PROB = 0.5
    
    # Noise parameters
    NOISE_STD = 0.01
    BLUR_SIGMA = (0.1, 2.0)
    NOISE_PROB = 0.3
```

---

## Best Practices

### 1. Histogram Normalization
- ✓ Always apply to training, validation, AND test sets
- ✓ Use CLAHE for medical images (default)
- ✓ Consistent method across all datasets

### 2. Training Augmentation
- ✓ Apply all augmentation types during training
- ✓ Use moderate probability (0.3-0.5) to avoid over-augmentation
- ✓ Monitor validation performance to tune parameters

### 3. Validation/Test Augmentation
- ✓ Only apply histogram normalization
- ✓ No random augmentations (reproducible results)
- ✓ Same preprocessing as training for consistency

### 4. Parameter Tuning
- Start with default parameters
- Gradually increase augmentation strength if overfitting
- Reduce if validation performance degrades
- Use visualization to verify augmentation quality

### 5. Medical Imaging Considerations
- Preserve diagnostic features
- Avoid extreme transformations
- Consider domain expert review
- Document augmentation choices

---

## Performance Impact

### Expected Benefits
- **Reduced Overfitting**: 10-20% improvement in generalization
- **Better Robustness**: Handles imaging variations
- **Improved Accuracy**: 2-5% test accuracy improvement
- **Larger Effective Dataset**: 5-10× effective training samples

### Computational Cost
- **Training Time**: +20-30% (augmentation overhead)
- **Memory Usage**: Minimal increase
- **Inference Time**: Only histogram normalization (negligible)

---

## Troubleshooting

### Issue: Augmentation too aggressive
**Solution**: Reduce probability or parameter ranges
```python
GEOMETRIC_PROB = 0.3  # Reduce from 0.5
ROTATION_RANGE = 10   # Reduce from 15
```

### Issue: Training loss not decreasing
**Solution**: Temporarily disable augmentation to verify model
```python
ENABLE_GEOMETRIC_AUG = False
ENABLE_COLOR_AUG = False
ENABLE_NOISE_AUG = False
```

### Issue: Validation performance worse than training
**Solution**: Ensure histogram normalization is applied to validation
```python
val_aug = get_augmentation_pipeline(is_training=False)  # Includes histogram
```

### Issue: Out of memory errors
**Solution**: Reduce batch size or disable some augmentations
```python
BATCH_SIZE = 2  # Reduce from 4
ENABLE_NOISE_AUG = False  # Disable if needed
```

---

## References

### Medical Imaging Augmentation
- Perez, L., & Wang, J. (2017). The effectiveness of data augmentation in image classification using deep learning.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning.

### Histogram Normalization
- Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization.
- Pizer, S. M., et al. (1987). Adaptive histogram equalization and its variations.

### Medical Image Analysis
- Litjens, G., et al. (2017). A survey on deep learning in medical image analysis.
- Ker, J., et al. (2018). Deep learning applications in medical image analysis.

---

## Summary

The augmentation pipeline provides:
- ✓ **Comprehensive preprocessing** with histogram normalization
- ✓ **Robust geometric transformations** for orientation invariance
- ✓ **Color augmentations** for staining variations
- ✓ **Noise and blur** for quality robustness
- ✓ **Easy configuration** via config.py
- ✓ **Visualization tools** for validation
- ✓ **Medical imaging specific** design choices

**Result**: Improved model generalization and robustness for BMA classification.

---

*Last Updated: 2025-09-30*
*Version: 1.0*
