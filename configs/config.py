"""
Configuration file for BMA MIL Classifier
"""

import torch


class Config:
    # Data parameters
    DATA_PATH = 'data/BWM_label_data.csv'
    IMAGE_DIR = 'data/images'  # Update this to your image directory
    NUM_CLASSES = 4

    # Image processing
    ORIGINAL_SIZE = (4032, 3024)
    PATCH_SIZE = 1008
    TARGET_SIZE = 224
    NUM_PATCHES_PER_IMAGE = 12
    MAX_IMAGES_PER_PILE = 50

    # Model architecture
    FEATURE_DIM = 768  # ViT-R50 feature dimension
    IMAGE_HIDDEN_DIM = 512
    PILE_HIDDEN_DIM = 256

    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.3
    
    # Class imbalance handling
    USE_WEIGHTED_LOSS = True

    # Data split
    TEST_SIZE = 0.3
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Cross-validation
    USE_CROSS_VALIDATION = False
    NUM_FOLDS = 3

    # Feature extractor
    FEATURE_EXTRACTOR_MODEL = 'vit_base_r50_s16_224.orig_in21k'

    # Paths
    BEST_MODEL_PATH = 'models/best_bma_mil_model.pth'
    TRAINING_PLOT_PATH = 'results/training_history.png'
    LOG_DIR = 'logs'

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Logging
    ENABLE_LOGGING = True
    LOG_LEVEL = 'INFO'
    
    # Data Augmentation Parameters
    HISTOGRAM_METHOD = 'clahe'
    
    # Enable/disable augmentation types
    ENABLE_GEOMETRIC_AUG = True
    ENABLE_COLOR_AUG = False
    ENABLE_NOISE_AUG = False
    
    # Geometric augmentation parameters
    ROTATION_RANGE = 15
    ZOOM_RANGE = (0.9, 1.1)
    SHEAR_RANGE = 10
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    GEOMETRIC_PROB = 0.5
    
    # Color augmentation parameters
    BRIGHTNESS_RANGE = (0.8, 1.2)
    CONTRAST_RANGE = (0.8, 1.2)
    SATURATION_RANGE = (0.8, 1.2)
    HUE_RANGE = (-0.1, 0.1)
    COLOR_PROB = 0.5
    
    # Noise and blur parameters
    NOISE_STD = 0.01
    BLUR_SIGMA = (0.1, 2.0)
    NOISE_PROB = 0.3
