"""
BMA MIL Classifier Neural Network Models
Architecture: Patch → Image → Pile level aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageLevelAggregator(nn.Module):
    """Aggregate patch features to image-level representation"""

    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Attention mechanism for patch aggregation
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Image-level feature transformation
        self.image_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, patch_features):
        """
        Args:
            patch_features: Tensor of shape [batch_size, num_patches, input_dim]
        Returns:
            image_features: Tensor of shape [batch_size, hidden_dim]
        """
        batch_size, num_patches, _ = patch_features.shape

        # Compute attention weights
        attention_weights = self.attention(patch_features.view(-1, self.input_dim))
        attention_weights = attention_weights.view(batch_size, num_patches)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum of patch features
        weighted_features = torch.sum(patch_features * attention_weights.unsqueeze(-1), dim=1)

        # Transform to image-level representation
        image_features = self.image_encoder(weighted_features)

        return image_features


class PileLevelAggregator(nn.Module):
    """Aggregate image features to pile-level representation"""

    def __init__(self, input_dim=512, hidden_dim=256, num_classes=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Attention mechanism for image aggregation
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Pile-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, image_features):
        """
        Args:
            image_features: Tensor of shape [batch_size, num_images, input_dim]
        Returns:
            pile_logits: Tensor of shape [batch_size, num_classes]
            attention_weights: Tensor of shape [batch_size, num_images]
        """
        batch_size, num_images, _ = image_features.shape

        # Compute attention weights
        attention_weights = self.attention(image_features.view(-1, self.input_dim))
        attention_weights = attention_weights.view(batch_size, num_images)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum of image features
        weighted_features = torch.sum(image_features * attention_weights.unsqueeze(-1), dim=1)

        # Final classification
        pile_logits = self.classifier(weighted_features)

        return pile_logits, attention_weights


class BMA_MIL_Classifier(nn.Module):
    """Complete BMA MIL Classifier: Patch → Image → Pile"""

    def __init__(self, feature_dim=768, image_hidden_dim=512, pile_hidden_dim=256, num_classes=4):
        super().__init__()

        self.image_aggregator = ImageLevelAggregator(
            input_dim=feature_dim,
            hidden_dim=image_hidden_dim
        )

        self.pile_aggregator = PileLevelAggregator(
            input_dim=image_hidden_dim,
            hidden_dim=pile_hidden_dim,
            num_classes=num_classes
        )

    def forward(self, patch_features_list):
        """
        Args:
            patch_features_list: List of tensors, each of shape [num_images, num_patches, feature_dim]
        Returns:
            pile_logits: Tensor of shape [batch_size, num_classes]
            image_attention_weights: List of attention weights for each pile
        """
        batch_size = len(patch_features_list)

        # Process each pile
        all_image_features = []

        for patch_features in patch_features_list:
            # Aggregate patches to image level
            image_features = self.image_aggregator(patch_features)
            all_image_features.append(image_features)

        # Stack image features for batch processing
        max_images = max([img_feat.shape[0] for img_feat in all_image_features])

        # Pad image features to same length
        padded_image_features = []
        device = all_image_features[0].device  # Get device from first tensor
        for img_feat in all_image_features:
            num_images = img_feat.shape[0]
            if num_images < max_images:
                padding = torch.zeros(max_images - num_images, img_feat.shape[1], device=device)
                padded = torch.cat([img_feat, padding], dim=0)
            else:
                padded = img_feat
            padded_image_features.append(padded)

        image_features_batch = torch.stack(padded_image_features, dim=0)

        # Aggregate to pile level
        pile_logits, pile_attention_weights = self.pile_aggregator(image_features_batch)

        return pile_logits, pile_attention_weights
