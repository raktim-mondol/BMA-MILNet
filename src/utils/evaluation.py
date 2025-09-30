"""
Model evaluation utilities
"""

import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_model(model, test_loader):
    """Evaluate the trained model"""
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_labels = []
    all_pile_names = []

    with torch.no_grad():
        for patch_features_list, labels, pile_names in test_loader:
            labels = labels.to(device)
            patch_features_device = [feat.to(device) for feat in patch_features_list]

            pile_logits, attention_weights = model(patch_features_device)
            preds = torch.argmax(pile_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_pile_names.extend(pile_names)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return accuracy, f1, cm, all_preds, all_labels, all_pile_names
