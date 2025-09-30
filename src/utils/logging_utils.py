"""
Logging and result saving utilities
"""

import os
import logging
from datetime import datetime


def setup_logging(log_dir='logs', mode='standard'):
    """Setup logging configuration"""
    from configs.config import Config
    
    if not Config.ENABLE_LOGGING:
        return None
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'bma_mil_{mode}_{timestamp}.log')
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Device: {Config.DEVICE}")
    
    return logger


def save_results_to_file(results_data, accuracy, f1_scores, confusion_matrix, mode='standard'):
    """Save training results to a text file"""
    from configs.config import Config
    
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(Config.LOG_DIR, f'results_{mode}_{timestamp}.txt')
    
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"BMA MIL Classifier - Results Summary\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Epochs: {Config.NUM_EPOCHS}\n")
        f.write(f"  Batch Size: {Config.BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {Config.LEARNING_RATE}\n")
        f.write(f"  Device: {Config.DEVICE}\n\n")
        
        if mode == 'cross_validation':
            f.write(f"Cross-Validation Results ({Config.NUM_FOLDS} Folds):\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            
            f.write("Per-Class F1 Scores:\n")
            for class_idx, f1 in enumerate(f1_scores):
                f.write(f"  BMA Class {class_idx + 1}: {f1:.4f}\n")
            
            f.write("\nAverage Confusion Matrix:\n")
            f.write("Predicted →\n")
            f.write("True ↓    BMA1   BMA2   BMA3   BMA4\n")
            for i in range(Config.NUM_CLASSES):
                row_str = f"BMA {i+1}:  "
                for j in range(Config.NUM_CLASSES):
                    row_str += f"{confusion_matrix[i, j]:5.1f}  "
                f.write(row_str + "\n")
        else:
            f.write("Test Set Results:\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"Weighted F1 Score: {results_data['weighted_f1']:.4f}\n\n")
            
            f.write("Per-Class F1 Scores:\n")
            for class_idx, f1 in enumerate(f1_scores):
                f.write(f"  BMA Class {class_idx + 1}: {f1:.4f}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write("Predicted →\n")
            f.write("True ↓    BMA1  BMA2  BMA3  BMA4\n")
            for i in range(Config.NUM_CLASSES):
                row_str = f"BMA {i+1}:  "
                for j in range(Config.NUM_CLASSES):
                    row_str += f"{confusion_matrix[i, j]:4d}  "
                f.write(row_str + "\n")
    
    print(f"\nResults saved to: {results_file}")
    if logging.getLogger(__name__).hasHandlers():
        logging.info(f"Results saved to: {results_file}")
