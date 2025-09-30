"""
Main training script for BMA MIL Classifier
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from src.models import BMA_MIL_Classifier
from src.data import BMADataset
from src.feature_extractor import FeatureExtractor
from src.augmentation import get_augmentation_pipeline
from src.utils import (
    train_model,
    compute_class_weights,
    evaluate_model,
    setup_logging,
    save_results_to_file
)
from configs.config import Config


def main():
    """Main training and evaluation pipeline"""
    
    # Setup logging
    mode = 'standard'
    logger = setup_logging(log_dir=Config.LOG_DIR, mode=mode)

    # Load data
    df = pd.read_csv(Config.DATA_PATH)
    df = df[df['BMA_label'] != 'BMA_label']
    df['BMA_label'] = df['BMA_label'].astype(int)

    print(f"\n{'='*60}")
    print(f"BMA MIL Classifier - Training Pipeline")
    print(f"{'='*60}")
    print(f"Device: {Config.DEVICE}")
    print(f"Early Stopping: {'Enabled' if Config.USE_EARLY_STOPPING else 'Disabled'}")
    print(f"Logging: {'Enabled' if Config.ENABLE_LOGGING else 'Disabled'}")
    print(f"{'='*60}\n")
    
    if logger:
        logger.info(f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, LR={Config.LEARNING_RATE}")
        logger.info(f"Augmentation: Geometric={Config.ENABLE_GEOMETRIC_AUG}, Color={Config.ENABLE_COLOR_AUG}, Noise={Config.ENABLE_NOISE_AUG}")

    # Split piles into train/val/test with stratification
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
    unique_piles = pile_labels['pile'].values
    pile_bma_labels = pile_labels['BMA_label'].values
    
    train_piles, temp_piles, train_labels, temp_labels = train_test_split(
        unique_piles, pile_bma_labels, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, 
        stratify=pile_bma_labels
    )
    val_piles, test_piles, val_labels, test_labels = train_test_split(
        temp_piles, temp_labels, 
        test_size=0.5, 
        random_state=Config.RANDOM_STATE, 
        stratify=temp_labels
    )

    train_df = df[df['pile'].isin(train_piles)]
    val_df = df[df['pile'].isin(val_piles)]
    test_df = df[df['pile'].isin(test_piles)]

    print(f"Training piles: {len(train_piles)}")
    print(f"Validation piles: {len(val_piles)}")
    print(f"Test piles: {len(test_piles)}")
    
    print("\nClass distribution in splits:")
    print(f"Train - BMA classes: {train_df.groupby('BMA_label')['pile'].nunique().to_dict()}")
    print(f"Val   - BMA classes: {val_df.groupby('BMA_label')['pile'].nunique().to_dict()}")
    print(f"Test  - BMA classes: {test_df.groupby('BMA_label')['pile'].nunique().to_dict()}")

    # Initialize feature extractor
    feature_extractor = FeatureExtractor()

    # Create augmentation pipelines
    print("\nInitializing data augmentation pipelines...")
    train_augmentation = get_augmentation_pipeline(is_training=True, config=Config)
    val_augmentation = get_augmentation_pipeline(is_training=False, config=Config)
    print("✓ Training augmentation: Histogram normalization + Geometric")
    print("✓ Validation/Test augmentation: Histogram normalization only")

    # Create datasets with augmentation
    train_dataset = BMADataset(train_df, Config.IMAGE_DIR, feature_extractor, 
                              augmentation=train_augmentation, is_training=True)
    val_dataset = BMADataset(val_df, Config.IMAGE_DIR, feature_extractor, 
                            augmentation=val_augmentation, is_training=False)
    test_dataset = BMADataset(test_df, Config.IMAGE_DIR, feature_extractor, 
                             augmentation=val_augmentation, is_training=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, collate_fn=lambda x: x)

    # Initialize model
    model = BMA_MIL_Classifier(
        feature_dim=Config.FEATURE_DIM,
        image_hidden_dim=Config.IMAGE_HIDDEN_DIM,
        pile_hidden_dim=Config.PILE_HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES
    )

    model = model.to(Config.DEVICE)

    print(f"Using device: {Config.DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Compute class weights if enabled
    class_weights = None
    if Config.USE_WEIGHTED_LOSS:
        class_weights = compute_class_weights(train_df, num_classes=Config.NUM_CLASSES, 
                                             device=Config.DEVICE)
    else:
        print("\nWeighted loss disabled - using standard loss")

    # Train model
    train_losses, val_accuracies, val_f1_scores = train_model(
        model, train_loader, val_loader,
        num_epochs=Config.NUM_EPOCHS, 
        learning_rate=Config.LEARNING_RATE,
        class_weights=class_weights
    )

    # Load best model and evaluate
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    accuracy, f1, cm, preds, labels, pile_names = evaluate_model(model, test_loader)
    
    # Calculate per-class F1 scores
    f1_per_class = f1_score(labels, preds, average=None)
    
    print(f"\n{'='*60}")
    print(f"Test Set Results:")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"\nPer-Class F1 Scores:")
    for class_idx, f1_score_val in enumerate(f1_per_class):
        print(f"  BMA Class {class_idx + 1}: {f1_score_val:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"{'='*60}")
    print("Predicted →")
    print(f"True ↓    BMA1  BMA2  BMA3  BMA4")
    for i in range(Config.NUM_CLASSES):
        row_str = f"BMA {i+1}:  "
        for j in range(Config.NUM_CLASSES):
            row_str += f"{cm[i, j]:4d}  "
        print(row_str)
    print(f"{'='*60}")

    # Plot training history
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Set Confusion Matrix')
    plt.colorbar(im)
    tick_marks = range(Config.NUM_CLASSES)
    plt.xticks(tick_marks, [f'BMA {i+1}' for i in range(Config.NUM_CLASSES)])
    plt.yticks(tick_marks, [f'BMA {i+1}' for i in range(Config.NUM_CLASSES)])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(Config.TRAINING_PLOT_PATH)
    print(f"\nTraining plots saved to '{Config.TRAINING_PLOT_PATH}'")

    print("\nTraining completed!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test F1 Score: {f1:.4f}")
    
    # Save results
    results_dict = {'weighted_f1': f1}
    save_results_to_file(results_dict, accuracy, f1_per_class, cm, mode='standard')


if __name__ == "__main__":
    main()
