"""
Material Classification Training Script - my attempt at scrap sorting
Trains ResNet18 on the trashnet data, should be decent for the assignment
Note: I used transfer learning cuz training from scratch takes forever on my machine
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

class MaterialDataset(Dataset):
    """Custom Dataset for Material Classification - loads images from folders"""
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load images and labels, ignore any broken files i guess
        self.samples = []
        self.class_names = sorted(os.listdir(os.path.join(data_dir, split)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        for class_name in self.class_names:
            class_path = os.path.join(data_dir, split, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(class_path, img_name),
                            'label': self.class_to_idx[class_name]
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Define data augmentation and preprocessing transforms - added some flips and stuff for better training"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def build_model(num_classes, pretrained=True):
    """Build ResNet18 model with transfer learning - froze some layers to save time"""
    model = models.resnet18(pretrained=pretrained)
    
    # Freeze early layers for transfer learning, dont want to train everything
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Replace final layer for our 6 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss = running_loss / len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_acc, all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(data_dir, num_epochs=20, batch_size=32, lr=0.001):
    """Main training function - this is where the magic happens lol"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = MaterialDataset(data_dir, transform=train_transform, split='train')
    val_dataset = MaterialDataset(data_dir, transform=val_transform, split='val')
    
    # Create dataloaders, num_workers=2 cuz more might crash on windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Get number of classes
    num_classes = len(train_dataset.class_names)
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {train_dataset.class_names}')
    
    # Build model
    model = build_model(num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.class_names,
                'class_to_idx': train_dataset.class_to_idx
            }, 'models/best_model.pth')
            print(f'Saved best model with val_acc: {val_acc:.4f}')
    
    # Final evaluation
    print('\n=== Final Evaluation ===')
    model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])
    _, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print(f'Final Accuracy: {val_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    # Save metrics
    metrics = {
        'accuracy': float(val_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, train_dataset.class_names, 'results/confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
    
    return model, train_dataset.class_names

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Material Classification Model - for the scrap sorting thing')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Create directories if not there
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Determine device - auto is best i think
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA not available, falling back to CPU - bummer')
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU name: {torch.cuda.get_device_name(0)} - nice!')
    
    # Train model
    data_dir = 'data/materials'
    model, class_names = train_model(data_dir, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    
    print('\nTraining completed! Hope the accuracy is good enough for the assignment')