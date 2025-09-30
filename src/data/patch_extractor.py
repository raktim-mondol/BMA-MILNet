"""
Patch extraction from full-resolution images
"""

from PIL import Image


class PatchExtractor:
    """Extract 12 patches from 4032x3024 images"""

    def __init__(self, patch_size=1008, target_size=224, augmentation=None):
        self.patch_size = patch_size
        self.target_size = target_size
        self.num_patches_per_image = 12
        self.augmentation = augmentation

    def extract_patches(self, image_path):
        """Extract 12 patches from a single image with optional augmentation"""
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')

            # Verify image dimensions
            if img.size != (4032, 3024):
                print(f"Warning: Image {image_path} has size {img.size}, expected (4032, 3024)")
                # Resize if needed
                img = img.resize((4032, 3024))

            patches = []

            # Extract 12 patches (3 rows × 4 columns)
            for row in range(3):
                for col in range(4):
                    left = col * self.patch_size
                    upper = row * self.patch_size
                    right = left + self.patch_size
                    lower = upper + self.patch_size

                    # Ensure we don't exceed image boundaries
                    if right <= 4032 and lower <= 3024:
                        patch = img.crop((left, upper, right, lower))
                        # Resize to target size for ViT
                        patch = patch.resize((self.target_size, self.target_size))
                        
                        # Apply augmentation if provided
                        if self.augmentation is not None:
                            patch = self.augmentation(patch)
                        
                        patches.append(patch)

            if len(patches) != self.num_patches_per_image:
                print(f"Warning: Extracted {len(patches)} patches from {image_path}, expected {self.num_patches_per_image}")

            return patches

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
