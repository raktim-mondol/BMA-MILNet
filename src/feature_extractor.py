"""
Feature extraction using ViT-R50 model
"""

import torch
import timm


class FeatureExtractor:
    """Feature extraction using ViT-R50 model"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = timm.create_model(
            'vit_base_r50_s16_224.orig_in21k',
            pretrained=True,
            num_classes=0  # Remove classifier
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Get model transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def extract_features(self, patches):
        """Extract features from list of PIL patches"""
        features = []

        with torch.no_grad():
            for patch in patches:
                # Apply transforms
                tensor_patch = self.transform(patch).unsqueeze(0).to(self.device)

                # Extract features
                feature = self.model(tensor_patch)
                features.append(feature.cpu())

        if features:
            return torch.cat(features, dim=0)  # Shape: [num_patches, feature_dim]
        else:
            return torch.tensor([])
