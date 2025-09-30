# BMA Classification using Multi-Level MIL - Implementation Summary

## 🎯 Problem Overview

**Objective**: Perform BMA classification (4 categories) using all images from the same pile with a multi-level Multiple Instance Learning (MIL) approach.

**Key Requirements**:
- Each image: 4032×3024 pixels → 12 patches (1008×1008) → scaled to 224×224
- Multiple images per pile → final prediction using all images from that pile
- Feature extraction using ViT-R50 hybrid model
- Multi-level aggregation: Patch → Image → Pile

## 📊 Dataset Analysis

- **Total piles**: 123
- **BMA labels**: 1, 2, 3, 4 (4 categories)
- **Label distribution**:
  - Label 2: 14,635 images (majority)
  - Label 3: 4,889 images
  - Label 1: 532 images
  - Label 4: 300 images
- **Variable images per pile**: Each pile contains different number of images

## 🏗️ Architecture Design

### Multi-Level MIL Pipeline

```
4032×3024 Image
    ↓ (12 patches)
1008×1008 Patches (12 per image)
    ↓ (Resize)
224×224 Patches
    ↓ (ViT-R50)
768-dimensional Features
    ↓ (Image-level Attention)
512-dimensional Image Representation
    ↓ (Pile-level Attention)
4-class BMA Prediction
```

### Key Components

1. **PatchExtractor**: Divides 4032×3024 images into 12 patches (1008×1008) and resizes to 224×224
2. **FeatureExtractor**: Uses ViT-R50 (`vit_base_r50_s16_224.orig_in21k`) for 768-dimensional feature extraction
3. **ImageLevelAggregator**: Attention mechanism to aggregate 12 patch features into single image representation
4. **PileLevelAggregator**: Attention mechanism to aggregate multiple image features into pile-level prediction
5. **BMA_MIL_Classifier**: Complete end-to-end model

## 🧠 Model Details

### Feature Extraction
- **Model**: ViT-R50 (`vit_base_r50_s16_224.orig_in21k`)
- **Input**: 224×224 RGB patches
- **Output**: 768-dimensional features per patch
- **Pre-trained**: Yes (ImageNet-21k)

### Attention Mechanisms
- **Image Level**: Learns importance weights for 12 patches per image
- **Pile Level**: Learns importance weights for multiple images per pile
- **Benefits**:
  - Handles variable number of images per pile
  - Focuses on most informative patches/images
  - Provides interpretability through attention weights

### Model Parameters
- **Feature dimension**: 768
- **Image hidden dimension**: 512
- **Pile hidden dimension**: 256
- **Number of classes**: 4
- **Total parameters**: ~2.5M

## 🚀 Implementation Features

### ✅ Tested Components

1. **Patch Extraction**: ✅ Successfully extracts 12 patches from 4032×3024 images
2. **Feature Extraction**: ✅ ViT-R50 produces 768-dimensional features
3. **Variable Input Handling**: ✅ Supports different numbers of images per pile
4. **Multi-level Attention**: ✅ Both image and pile level attention mechanisms working
5. **End-to-end Pipeline**: ✅ Complete forward pass tested with dummy data

### 🛠️ Code Structure

```
├── bma_mil_classifier.py      # Main implementation
├── simple_demo.py             # Simplified architecture test
├── test_with_dummy_images.py  # Complete pipeline test
├── config.py                  # Configuration parameters
├── requirements.txt           # Dependencies
└── IMPLEMENTATION_SUMMARY.md  # This file
```

## 🎯 Training Pipeline

### Data Preparation
- **Train/Val/Test Split**: 70%/15%/15% of piles
- **Batch Processing**: Handles variable images per pile
- **Feature Caching**: Optional feature pre-extraction for faster training

### Training Configuration
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 1e-4
- **Batch Size**: 4 (piles per batch)
- **Epochs**: 50
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Based on validation accuracy

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 for imbalanced classes
- **Confusion Matrix**: Per-class performance
- **Attention Visualization**: Interpretability analysis

## 🔧 Usage Instructions

### 1. Setup Environment
```bash
source /home/raktim/upython/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
- Update `IMAGE_DIR` in `config.py` to point to your image directory
- Ensure CSV file `BWM_label_data.csv` is in the working directory

### 3. Run Training
```bash
python bma_mil_classifier.py
```

### 4. Monitor Progress
- Training loss and validation accuracy plots
- Best model saved as `best_bma_mil_model.pth`
- Test evaluation results printed

## 🎨 Key Innovations

1. **Hierarchical MIL**: First patch-level then image-level aggregation
2. **Variable Input Support**: Handles different numbers of images per pile
3. **Attention Mechanisms**: Learns importance weights at multiple levels
4. **Pre-trained Features**: Leverages ViT-R50 for robust feature extraction
5. **Interpretability**: Attention weights show which patches/images contribute most

## 📈 Expected Performance

Given the architecture and dataset characteristics:
- **Strong baseline**: Expected accuracy > 80% on test set
- **Robust features**: ViT-R50 provides excellent feature representation
- **Attention benefits**: Should outperform simple averaging approaches
- **Scalability**: Can handle large datasets efficiently

## 🚀 Next Steps

1. **Run on actual data**: Update image directory path and run full training
2. **Hyperparameter tuning**: Optimize learning rate, hidden dimensions
3. **Advanced techniques**: Add data augmentation, regularization
4. **Interpretability**: Analyze attention patterns for clinical insights
5. **Deployment**: Package model for inference on new piles

## 📚 References

- **ViT-R50**: [timm library](https://github.com/rwightman/pytorch-image-models)
- **MIL with Attention**: [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712)
- **Histopathology AI**: Recent advances in computational pathology

---

**Status**: ✅ Implementation complete and tested with dummy data
**Ready for**: Training on actual BMA dataset