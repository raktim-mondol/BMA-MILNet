"""
Training utilities
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from .early_stopping import EarlyStopping


def compute_class_weights(train_df, num_classes=4, device='cpu'):
    """Compute class weights based on inverse frequency for handling class imbalance"""
    pile_labels = train_df.groupby('pile')['BMA_label'].first().values
    pile_labels_indexed = pile_labels - 1
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=pile_labels_indexed
    )
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print("\nClass weights for handling imbalance:")
    for i, weight in enumerate(class_weights):
        class_count = np.sum(pile_labels_indexed == i)
        print(f"  BMA Class {i+1}: weight={weight:.4f}, count={class_count} piles")
    
    return class_weights_tensor


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, class_weights=None, fold=None):
    """Train the BMA MIL classifier"""
    from configs.config import Config
    
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        msg = "Using weighted CrossEntropyLoss for class imbalance"
        print(msg)
        if logger.hasHandlers():
            logger.info(msg)
    else:
        criterion = nn.CrossEntropyLoss()
        msg = "Using standard CrossEntropyLoss"
        print(msg)
        if logger.hasHandlers():
            logger.info(msg)

    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    best_val_acc = 0.0
    
    early_stopping = None
    if Config.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            verbose=True
        )
        msg = f"Early stopping enabled: patience={Config.EARLY_STOPPING_PATIENCE}, min_delta={Config.EARLY_STOPPING_MIN_DELTA}"
        print(msg)
        if logger.hasHandlers():
            logger.info(msg)

    fold_str = f" (Fold {fold})" if fold is not None else ""
    
    # Overall epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc=f'Training Progress{fold_str}', unit='epoch')
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        epoch_loss = 0.0

        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - Training', 
                         leave=False, unit='batch')
        
        for batch_idx, (patch_features_list, labels, pile_names) in enumerate(train_pbar):
            optimizer.zero_grad()

            labels = labels.to(device)
            patch_features_device = [feat.to(device) for feat in patch_features_list]

            pile_logits, _ = model(patch_features_device)
            loss = criterion(pile_logits, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Update progress bar with current loss
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}'})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        # Validation progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - Validation', 
                       leave=False, unit='batch')

        with torch.no_grad():
            for patch_features_list, labels, pile_names in val_pbar:
                labels = labels.to(device)
                patch_features_device = [feat.to(device) for feat in patch_features_list]

                pile_logits, _ = model(patch_features_device)
                preds = torch.argmax(pile_logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)

        # Update overall epoch progress bar with metrics
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}',
            'Val F1': f'{val_f1:.4f}',
            'Best Acc': f'{best_val_acc:.4f}'
        })

        msg = f'Epoch {epoch+1}/{num_epochs}{fold_str}, Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}'
        if logger.hasHandlers():
            logger.info(msg)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Config.BEST_MODEL_PATH if fold is None else f'best_bma_mil_model_fold{fold}.pth'
            torch.save(model.state_dict(), model_path)
            msg = f'New best model saved with validation accuracy: {best_val_acc:.4f}'
            print(msg)
            if logger.hasHandlers():
                logger.info(msg)
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_acc, epoch + 1):
                msg = f'Early stopping triggered at epoch {epoch+1}{fold_str}. Best accuracy: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}'
                print(msg)
                if logger.hasHandlers():
                    logger.info(msg)
                break

    return train_losses, val_accuracies, val_f1_scores
