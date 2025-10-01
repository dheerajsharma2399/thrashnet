"""
Dataset Preparation Script - gets the data ready for training
Handles TrashNet download and splitting, i made it auto for the resized folder
Took some time to get the splits right
"""

import os
import shutil
from pathlib import Path
import requests
import zipfile
from sklearn.model_selection import train_test_split
import json

def download_file(url, destination):
    """Download file from URL - with progress bar kinda"""
    print(f'Downloading from {url}...')
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                f.write(chunk)
                done = int(50 * downloaded / total_size)
                print(f'\r[{"=" * done}{" " * (50-done)}] {downloaded}/{total_size} bytes', end='')
    print()  # newline after progress


def prepare_trashnet_dataset():
    """
    Prepare TrashNet dataset - but i skipped download cuz its manual
    Has 6 classes: cardboard glass metal paper plastic trash
    """
    print('Preparing TrashNet dataset...')
    
    # Create dirs
    data_dir = Path('data/materials')
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Manual download note, couldnt automate it easily
    print('\nPlease download TrashNet manually:')
    print('1. Go to: https://github.com/garythung/trashnet/')
    print('2. Download and extract to: data/materials/raw/')
    print('3. Re-run this script after')
    
    # Check if its there
    dataset_path = raw_dir / 'dataset-resized'
    if not dataset_path.exists():
        print('\nDataset not found. Follow the steps above please.')
        return False
    
    return True

def merge_datasets(trashnet_dir, garbage_dir, combined_dir):
    """
    Merge TrashNet and Garbage V2 datasets - copy images per class to combined
    Maps Garbage V2 classes to TrashNet: battery/biological/clothes/e-waste/shoes -> trash
    Avoids duplicates by skipping same filename
    """
    print(f'Merging datasets from {trashnet_dir} and {garbage_dir} into {combined_dir}...')
    
    combined_path = Path(combined_dir)
    combined_path.mkdir(parents=True, exist_ok=True)
    
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Mapping for Garbage V2 classes to our classes
    garbage_mapping = {
        'cardboard': 'cardboard',
        'glass': 'glass',
        'metal': 'metal',
        'paper': 'paper',
        'plastic': 'plastic',
        'battery': 'trash',
        'biological': 'trash',
        'clothes': 'trash',
        'e-waste': 'trash',
        'shoes': 'trash'
    }
    
    for cls in classes:
        cls_dir = combined_path / cls
        cls_dir.mkdir(exist_ok=True)
        
        # Copy from TrashNet (direct match)
        trash_cls = Path(trashnet_dir) / cls
        if trash_cls.exists():
            for img in trash_cls.glob('*.jpg'):
                dest = cls_dir / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
        
        # Copy from Garbage V2 (map classes)
        for g_class, target_cls in garbage_mapping.items():
            if target_cls == cls:
                garbage_cls = Path(garbage_dir) / g_class
                if garbage_cls.exists():
                    for img in garbage_cls.glob('*.jpg'):
                        dest = cls_dir / img.name
                        if not dest.exists():
                            shutil.copy2(img, dest)
        
        print(f'{cls}: {len(list(cls_dir.glob("*.jpg")))} images')
    
    print(f'Merged dataset ready at {combined_dir} - all non-matching Garbage classes mapped to trash')
    return True

def prepare_custom_dataset(source_dir):
    """
    Prepare custom dataset from folder - if you have your own images
    Structure: source_dir/class1/img.jpg etc
    """
    print(f'Preparing custom dataset from {source_dir}...')
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f'Source directory not found: {source_dir} - check path')
        return False
    
    # Get class dirs
    classes = [d for d in source_path.iterdir() if d.is_dir()]
    
    if len(classes) < 5:
        print(f'Warning: Only {len(classes)} classes found. Need at least 5 for good model.')
        return False
    
    print(f'Found {len(classes)} classes: {[c.name for c in classes]}')
    return True

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train val test - using sklearn split
    Args:
        source_dir: where the class folders are
        output_dir: where to put the splits
        ratios for each
    """
    print(f'\nSplitting dataset...')
    print(f'Train: {train_ratio*100}%, Val: {val_ratio*100}%, Test: {test_ratio*100}%')
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Make output dirs
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Process classes
    class_stats = {}
    
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f'\nProcessing class: {class_name}')
        
        # Find images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(class_dir.glob(ext))
            images.extend(class_dir.glob(ext.upper()))
        
        if len(images) == 0:
            print(f'  No images in {class_name} - skip')
            continue
        
        print(f'  Found {len(images)} images')
        
        # Split em
        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, 
                                               train_size=val_ratio/(val_ratio+test_ratio), 
                                               random_state=42)
        
        class_stats[class_name] = {
            'total': len(images),
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs)
        }
        
        # Copy to splits
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            split_class_dir = output_path / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in imgs:
                dest_path = split_class_dir / img_path.name
                shutil.copy2(img_path, dest_path)
        
        print(f'  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}')
    
    # Save stats
    stats_file = output_path / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(class_stats, f, indent=4)
    
    # Class info
    class_names = sorted(class_stats.keys())
    class_info = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'class_to_idx': {cls: idx for idx, cls in enumerate(class_names)}
    }
    
    class_file = output_path / 'class_names.json'
    with open(class_file, 'w') as f:
        json.dump(class_info, f, indent=4)
    
    print(f'\n{"="*50}')
    print('Dataset Split Summary:')
    print(f'{"="*50}')
    for class_name, stats in class_stats.items():
        print(f'{class_name:15s} - Total: {stats["total"]:4d} | '
              f'Train: {stats["train"]:4d} | Val: {stats["val"]:3d} | Test: {stats["test"]:3d}')
    
    print(f'\nDataset prepared successfully!')
    print(f'Output directory: {output_dir}')
    print(f'Class names saved to: {class_file}')
    print(f'Statistics saved to: {stats_file}')
    
    return True

def create_sample_dataset():
    """Create a sample dataset structure for testing - if no real data"""
    print('Creating sample dataset structure...')
    
    base_dir = Path('data/materials/sample')
    classes = ['metal', 'plastic', 'paper', 'glass', 'cardboard']
    
    for split in ['train', 'val', 'test']:
        for cls in classes:
            class_dir = base_dir / split / cls
            class_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Sample dataset structure created at: {base_dir}')
    print('Add your images to the class folders manually')

def main():
    print('='*60)
    print('Dataset Preparation Script - Auto mode for TrashNet + Garbage V2')
    print('='*60)
    
    trashnet_dir = 'data/dataset-resized'
    garbage_dir = 'data/garbage-v2'
    combined_dir = 'data/combined'
    output_dir = 'data/materials'
    
    # Merge if garbage v2 exists
    if Path(garbage_dir).exists():
        print(f'Merging Garbage V2 with TrashNet...')
        merge_datasets(trashnet_dir, garbage_dir, combined_dir)
        source_dir = combined_dir
    elif Path(trashnet_dir).exists():
        print(f'Using only TrashNet at: {trashnet_dir}')
        source_dir = trashnet_dir
    else:
        print('No dataset found. Download TrashNet to data/dataset-resized or Garbage V2 to data/garbage-v2.')
        return
    
    print(f'Splitting {source_dir} into: {output_dir}')
    success = split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    if success:
        print(f'\nDataset prepared successfully at {output_dir}! Larger dataset for better training.')
    else:
        print('Dataset preparation failed - check the logs')

if __name__ == '__main__':
    main()