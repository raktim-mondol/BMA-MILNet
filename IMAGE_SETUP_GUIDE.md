# Image Location Setup Guide

This guide explains how to configure the image directory for the BMA MIL Classifier.

## Overview

All your images should be stored in a **single folder**. The CSV file (`BWM_label_data.csv`) contains only the image filenames (e.g., "B1 (103).JPG"), and the `IMAGE_DIR` configuration tells the system where to find these images.

## Configuration Options

Edit the `IMAGE_DIR` parameter in `configs/config.py`:

### Option 1: Relative Path (Default)
```python
IMAGE_DIR = 'data/images'
```
Images should be in: `<project_root>/data/images/`

### Option 2: Absolute Windows Path (Recommended for Flexibility)

**Using raw string (recommended):**
```python
IMAGE_DIR = r'C:\Users\YourName\Pictures\pile_images'
```

**Using forward slashes:**
```python
IMAGE_DIR = 'C:/Users/YourName/Pictures/pile_images'
```

**Using escaped backslashes:**
```python
IMAGE_DIR = 'C:\\Users\\YourName\\Pictures\\pile_images'
```

## Example Setup

### Scenario: All images are in `D:\MyData\BMA_Images`

1. Open `configs/config.py`
2. Find the `IMAGE_DIR` line (around line 16)
3. Update it to:
   ```python
   IMAGE_DIR = r'D:\MyData\BMA_Images'
   ```

### Folder Structure Example

```
D:\MyData\BMA_Images\
├── B1 (103).JPG
├── B1 (106).JPG
├── B1 (107).JPG
├── B2 (45).JPG
├── B2 (46).JPG
└── ... (all other images)
```

Your CSV file should contain just the filenames:
```csv
Sl,pile,image_path,BMA_label
0,B1,B1 (103).JPG,2
1,B1,B1 (106).JPG,2
2,B1,B1 (107).JPG,2
```

## Important Notes

1. **All images must be in ONE folder** - the system will look for all images in the `IMAGE_DIR` location
2. **Image filenames in CSV must match exactly** - including spaces, parentheses, and file extensions
3. **Windows paths are fully supported** - use raw strings (`r'...'`) to avoid backslash issues
4. **No subfolders** - all images should be directly in the `IMAGE_DIR`, not in subdirectories

## Troubleshooting

**Problem:** Images not found during training

**Solutions:**
- Verify the path is correct and the folder exists
- Check that image filenames in CSV match the actual files (case-sensitive on some systems)
- Ensure you're using raw string notation for Windows paths: `r'C:\path\to\images'`
- Verify all images are in the single folder, not in subfolders

**Problem:** Path with spaces not working

**Solution:** Use raw string notation:
```python
IMAGE_DIR = r'C:\Users\John Doe\My Documents\Images'
```
