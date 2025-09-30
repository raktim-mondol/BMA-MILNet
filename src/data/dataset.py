"""
PyTorch Dataset for BMA classification
"""

import os
import torch
from torch.utils.data import Dataset
from .patch_extractor import PatchExtractor


class BMADataset(Dataset):
    """Dataset for BMA classification with pile-level aggregation"""

    def __init__(self, data_df, image_dir, feature_extractor=None, max_images_per_pile=50, 
                 augmentation=None, is_training=True):
        self.data_df = data_df
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.max_images_per_pile = max_images_per_pile
        self.is_training = is_training
        
        # Initialize patch extractor with augmentation
        self.patch_extractor = PatchExtractor(augmentation=augmentation)

        # Group by pile
        self.pile_groups = {}
        for pile_name, group in data_df.groupby('pile'):
            self.pile_groups[pile_name] = {
                'image_paths': group['image_path'].tolist()[:max_images_per_pile],
                'label': group['BMA_label'].iloc[0] - 1  # Convert to 0-indexed
            }

        self.pile_names = list(self.pile_groups.keys())

    def __len__(self):
        return len(self.pile_names)

    def __getitem__(self, idx):
        pile_name = self.pile_names[idx]
        pile_data = self.pile_groups[pile_name]

        image_paths = pile_data['image_paths']
        label = pile_data['label']

        # Extract patches and features for all images in pile
        all_patch_features = []

        for img_path in image_paths:
            full_path = os.path.join(self.image_dir, img_path)

            if os.path.exists(full_path):
                # Extract patches
                patches = self.patch_extractor.extract_patches(full_path)

                if patches and self.feature_extractor:
                    # Extract features
                    patch_features = self.feature_extractor.extract_features(patches)
                    if patch_features.numel() > 0:
                        all_patch_features.append(patch_features)

        if all_patch_features:
            # Stack patch features for this pile
            patch_features_tensor = torch.stack(all_patch_features, dim=0)  # [num_images, num_patches, feature_dim]
        else:
            # Create empty tensor if no features extracted
            patch_features_tensor = torch.zeros(1, 12, 768)

        return patch_features_tensor, label, pile_name
