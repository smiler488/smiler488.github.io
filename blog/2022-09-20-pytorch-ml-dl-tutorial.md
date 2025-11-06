---
slug: pytorch-ml-dl-tutorial
title: PyTorch Tutorial
authors: [liangchao]
tags: [pytorch, machine learning, deep learning, neural networks, tutorial]
---

<!-- truncate -->

# Complete PyTorch Tutorial for Machine Learning and Deep Learning

## Introduction

PyTorch is one of the most popular deep learning frameworks, known for its dynamic computation graphs, intuitive API, and strong community support. This comprehensive tutorial covers everything from basic tensor operations to advanced deep learning architectures, providing practical examples and best practices for both machine learning and deep learning applications.

## Table of Contents

1. [PyTorch Fundamentals](#pytorch-fundamentals)
2. [Data Handling and Preprocessing](#data-handling-and-preprocessing)
3. [Building Neural Networks](#building-neural-networks)
4. [Training and Optimization](#training-and-optimization)
5. [Computer Vision with PyTorch](#computer-vision-with-pytorch)
6. [Natural Language Processing](#natural-language-processing)
7. [Advanced Topics](#advanced-topics)
8. [Production Deployment](#production-deployment)

## PyTorch Fundamentals

### Installation and Setup

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Additional packages
pip install numpy matplotlib scikit-learn pandas seaborn jupyter
```

### Basic Tensor Operations

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Check PyTorch version and CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Creating tensors
def tensor_basics():
    # Different ways to create tensors
    x1 = torch.tensor([1, 2, 3, 4, 5])
    x2 = torch.zeros(3, 4)
    x3 = torch.ones(2, 3)
    x4 = torch.randn(2, 3)  # Random normal distribution
    x5 = torch.arange(0, 10, 2)  # Range tensor
    
    print("Basic tensor creation:")
    print(f"x1: {x1}")
    print(f"x2 shape: {x2.shape}")
    print(f"x4: {x4}")
    
    # Tensor properties
    print(f"\nTensor properties:")
    print(f"Data type: {x4.dtype}")
    print(f"Device: {x4.device}")
    print(f"Shape: {x4.shape}")
    print(f"Number of dimensions: {x4.ndim}")
    
    # Moving tensors to GPU
    if torch.cuda.is_available():
        x4_gpu = x4.cuda()
        print(f"GPU tensor device: {x4_gpu.device}")
    
    return x1, x2, x3, x4, x5

# Tensor operations
def tensor_operations():
    # Basic arithmetic operations
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    # Element-wise operations
    add_result = a + b
    mul_result = a * b
    div_result = a / b
    
    # Matrix operations
    matmul_result = torch.matmul(a, b)
    transpose_result = a.t()
    
    print("Tensor operations:")
    print(f"Addition: \n{add_result}")
    print(f"Matrix multiplication: \n{matmul_result}")
    print(f"Transpose: \n{transpose_result}")
    
    # Reshaping and indexing
    x = torch.randn(4, 6)
    x_reshaped = x.view(2, 12)  # Reshape
    x_slice = x[:2, :3]  # Slicing
    
    print(f"\nOriginal shape: {x.shape}")
    print(f"Reshaped: {x_reshaped.shape}")
    print(f"Sliced: {x_slice.shape}")
    
    return a, b, add_result, matmul_result

# Automatic differentiation
def autograd_example():
    # Enable gradient computation
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    # Forward pass
    z = x**2 + y**3
    
    # Backward pass
    z.backward()
    
    print("Automatic differentiation:")
    print(f"x.grad: {x.grad}")  # dz/dx = 2x = 4
    print(f"y.grad: {y.grad}")  # dz/dy = 3y^2 = 27
    
    # More complex example
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    
    print(f"\nFinal y: {y}")
    
    # Compute gradients
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)
    print(f"x.grad: {x.grad}")

# Run basic examples
tensor_basics()
tensor_operations()
autograd_example()
```

## Data Handling and Preprocessing

### Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target

# Image dataset example
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Assuming directory structure: root_dir/class_name/image.jpg
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data preprocessing and augmentation
def create_data_loaders():
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    # For demonstration, using CIFAR-10
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Data exploration
def explore_data(data_loader):
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Data type: {images.dtype}")
    print(f"Label range: {labels.min()} to {labels.max()}")
    
    # Visualize some samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        row, col = i // 4, i % 4
        img = images[i].permute(1, 2, 0)  # Change from CHW to HWC
        # Denormalize for visualization
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Label: {labels[i].item()}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Create and explore data
train_loader, val_loader = create_data_loaders()
explore_data(train_loader)
```

## Building Neural Networks

### Basic Neural Network Components

```python
# Simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ResNet-like block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

# Custom ResNet
class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Model initialization and summary
def initialize_model(model_type='cnn', num_classes=10):
    if model_type == 'simple':
        model = SimpleNN(input_size=32*32*3, hidden_size=512, num_classes=num_classes)
    elif model_type == 'cnn':
        model = CNN(num_classes=num_classes)
    elif model_type == 'resnet':
        model = CustomResNet(num_classes=num_classes)
    else:
        raise ValueError("Unknown model type")
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    model.apply(init_weights)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

# Test model creation
model = initialize_model('resnet')
print(model)
```

## Training and Optimization

### Training Loop Implementation

```python
import time
from tqdm import tqdm
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(len(pbar.iterable)):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs, scheduler=None, early_stopping_patience=None):
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            if scheduler:
                scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if early_stopping_patience:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        break
        
        print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%')
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Advanced optimization techniques
def setup_training(model, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with different options
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Alternative schedulers
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    
    return trainer, scheduler

# Mixed precision training
class MixedPrecisionTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training (Mixed Precision)')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

# Example training setup and execution
model = initialize_model('resnet', num_classes=10)
trainer, scheduler = setup_training(model, train_loader, val_loader)

# Train the model
trainer.train(num_epochs=50, scheduler=scheduler, early_stopping_patience=10)

# Plot training history
trainer.plot_training_history()
```

## Computer Vision with PyTorch

### Transfer Learning and Fine-tuning

```python
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# Transfer learning with pre-trained models
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'vit':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for feature extraction"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        if hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.backbone, 'classifier'):
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

# Object detection with YOLO-style architecture
class SimpleYOLO(nn.Module):
    def __init__(self, num_classes, num_anchors=3):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final FC layer
        
        # Detection head
        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, num_anchors * (5 + num_classes), 1)
        
    def forward(self, x):
        # Extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Detection head
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x

# Semantic segmentation with U-Net
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.final(dec1))

# Image augmentation and preprocessing
class AdvancedAugmentation:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Example: Fine-tuning a pre-trained model
def fine_tune_model():
    # Create transfer learning model
    model = TransferLearningModel(num_classes=10, model_name='resnet50', pretrained=True)
    
    # Phase 1: Feature extraction (freeze backbone)
    model.freeze_backbone()
    
    # Setup optimizer for feature extraction
    optimizer = optim.Adam(model.backbone.fc.parameters(), lr=0.001)
    
    print("Phase 1: Feature extraction training")
    # Train for a few epochs...
    
    # Phase 2: Fine-tuning (unfreeze backbone)
    model.unfreeze_backbone()
    
    # Setup optimizer for fine-tuning with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print("Phase 2: Fine-tuning training")
    # Continue training...
    
    return model

# Test computer vision models
transfer_model = fine_tune_model()
unet_model = UNet(in_channels=3, num_classes=21)  # For Pascal VOC
yolo_model = SimpleYOLO(num_classes=80)  # For COCO

print(f"Transfer learning model parameters: {sum(p.numel() for p in transfer_model.parameters()):,}")
print(f"U-Net model parameters: {sum(p.numel() for p in unet_model.parameters()):,}")
print(f"YOLO model parameters: {sum(p.numel() for p in yolo_model.parameters()):,}")
```

## Natural Language Processing

### Text Processing and RNN/Transformer Models

```python
import torch.nn.utils.rnn as rnn_utils
from collections import Counter
import re

# Text preprocessing utilities
class TextPreprocessor:
    def __init__(self, vocab_size=10000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
    
    def build_vocab(self, texts):
        # Tokenize and count words
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Build vocabulary
        vocab_words = [word for word, count in word_counts.most_common(self.vocab_size-4) 
                      if count >= self.min_freq]
        
        # Special tokens
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.vocab = set(special_tokens + vocab_words)
        
        # Create mappings
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def tokenize(self, text):
        # Simple tokenization
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def text_to_indices(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
    
    def indices_to_text(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

# LSTM-based language model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Output projection
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

# Transformer-based model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Embedding + positional encoding
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        # Transformer
        if mask is None:
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        output = self.transformer(x, mask)
        output = self.fc(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

# Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights

# Text classification model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers=2):
        super(TextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        
        # Classification
        output = self.fc(hidden)
        
        return output

# Example usage
def train_nlp_model():
    # Sample data
    texts = [
        "This is a sample sentence for training.",
        "Natural language processing with PyTorch is powerful.",
        "Deep learning models can understand text patterns."
    ]
    
    # Preprocess text
    preprocessor = TextPreprocessor(vocab_size=1000)
    preprocessor.build_vocab(texts)
    
    # Create models
    vocab_size = len(preprocessor.vocab)
    
    # LSTM Language Model
    lstm_model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_size=128,
        hidden_size=256,
        num_layers=2
    )
    
    # Transformer Model
    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        max_seq_len=100
    )
    
    # Text Classifier
    classifier_model = TextClassifier(
        vocab_size=vocab_size,
        embed_size=128,
        hidden_size=256,
        num_classes=3
    )
    
    print(f"LSTM model parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    print(f"Transformer model parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    print(f"Classifier model parameters: {sum(p.numel() for p in classifier_model.parameters()):,}")
    
    return lstm_model, transformer_model, classifier_model

# Test NLP models
lstm_model, transformer_model, classifier_model = train_nlp_model()
```

## Advanced Topics

### Custom Loss Functions and Metrics

```python
# Custom loss functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                    label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Custom metrics
class MetricsCalculator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.total_samples = 0
        self.correct_predictions = 0
    
    def update(self, predictions, targets):
        _, predicted = torch.max(predictions, 1)
        self.total_samples += targets.size(0)
        self.correct_predictions += (predicted == targets).sum().item()
        
        # Update confusion matrix
        for t, p in zip(targets.view(-1), predicted.view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1
    
    def accuracy(self):
        return self.correct_predictions / self.total_samples
    
    def precision(self, class_idx=None):
        if class_idx is not None:
            tp = self.confusion_matrix[class_idx, class_idx]
            fp = self.confusion_matrix[:, class_idx].sum() - tp
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        else:
            precisions = []
            for i in range(self.num_classes):
                precisions.append(self.precision(i))
            return sum(precisions) / len(precisions)
    
    def recall(self, class_idx=None):
        if class_idx is not None:
            tp = self.confusion_matrix[class_idx, class_idx]
            fn = self.confusion_matrix[class_idx, :].sum() - tp
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            recalls = []
            for i in range(self.num_classes):
                recalls.append(self.recall(i))
            return sum(recalls) / len(recalls)
    
    def f1_score(self, class_idx=None):
        if class_idx is not None:
            p = self.precision(class_idx)
            r = self.recall(class_idx)
            return 2 * p * r / (p + r) if (p + r) > 0 else 0
        else:
            f1_scores = []
            for i in range(self.num_classes):
                f1_scores.append(self.f1_score(i))
            return sum(f1_scores) / len(f1_scores)

# Model interpretability
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx):
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam / torch.max(cam)
        
        return cam

# Regularization techniques
class DropBlock2D(nn.Module):
    def __init__(self, drop_rate, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
    
    def forward(self, x):
        if not self.training:
            return x
        
        gamma = self.drop_rate / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        
        # Expand mask
        mask = mask.unsqueeze(1)
        mask = F.max_pool2d(mask, (self.block_size, self.block_size), 
                           stride=(1, 1), padding=self.block_size // 2)
        
        mask = 1 - mask
        normalize_factor = mask.numel() / mask.sum()
        
        return x * mask * normalize_factor

class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

# Knowledge distillation
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs, teacher_outputs, targets):
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = self.kl_div(soft_student, soft_targets) * (self.temperature ** 2)
        
        # Student loss
        student_loss = self.ce_loss(student_outputs, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss

# Example usage of advanced techniques
def advanced_training_example():
    # Create model
    model = CNN(num_classes=10)
    
    # Custom loss
    focal_loss = FocalLoss(alpha=1, gamma=2)
    
    # Metrics calculator
    metrics = MetricsCalculator(num_classes=10)
    
    # MixUp augmentation
    mixup = MixUp(alpha=1.0)
    
    # Training loop with advanced techniques
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Apply MixUp
        mixed_data, target_a, target_b, lam = mixup(data, target)
        
        # Forward pass
        output = model(mixed_data)
        
        # Calculate loss with MixUp
        loss = lam * focal_loss(output, target_a) + (1 - lam) * focal_loss(output, target_b)
        
        # Update metrics
        metrics.update(output, target)
        
        # Backward pass
        loss.backward()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            print(f'Accuracy: {metrics.accuracy():.4f}')
            print(f'F1 Score: {metrics.f1_score():.4f}')

# Test advanced techniques
advanced_training_example()
```

## Production Deployment

### Model Optimization and Deployment

```python
# Model optimization techniques
def optimize_model_for_inference(model, example_input):
    """Optimize model for inference"""
    
    # 1. Set to evaluation mode
    model.eval()
    
    # 2. Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # 3. Optimize for inference
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # 4. Save optimized model
    traced_model.save('optimized_model.pt')
    
    return traced_model

# Quantization
def quantize_model(model, data_loader):
    """Apply post-training quantization"""
    
    # Prepare model for quantization
    model.eval()
    model_fp32 = model
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model
    model_fp32_prepared = torch.quantization.prepare(model_fp32)
    
    # Calibrate with representative data
    with torch.no_grad():
        for data, _ in data_loader:
            model_fp32_prepared(data)
            break  # Use only one batch for calibration
    
    # Convert to quantized model
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    
    return model_int8

# ONNX export
def export_to_onnx(model, example_input, onnx_path):
    """Export model to ONNX format"""
    
    model.eval()
    
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {onnx_path}")

# TensorRT optimization (requires TensorRT)
def optimize_with_tensorrt(onnx_path, trt_path):
    """Optimize ONNX model with TensorRT"""
    try:
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision
        
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {trt_path}")
        return engine
        
    except ImportError:
        print("TensorRT not available")
        return None

# Model serving with Flask
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io

class ModelServer:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        
        # Define preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        # Preprocess image
        if isinstance(image, str):  # Base64 encoded
            image_data = base64.b64decode(image)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }

# Flask app
app = Flask(__name__)
model_server = ModelServer('optimized_model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        
        result = model_server.predict(image_data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

# Batch inference optimization
class BatchInferenceEngine:
    def __init__(self, model_path, batch_size=32, device='cuda'):
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.pending_requests = []
    
    def add_request(self, image, request_id):
        self.pending_requests.append((image, request_id))
        
        if len(self.pending_requests) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self):
        if not self.pending_requests:
            return []
        
        # Prepare batch
        images = []
        request_ids = []
        
        for image, request_id in self.pending_requests:
            images.append(image)
            request_ids.append(request_id)
        
        # Convert to tensor
        batch_tensor = torch.stack(images).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Prepare results
        results = []
        for i, request_id in enumerate(request_ids):
            predicted_class = torch.argmax(probabilities[i]).item()
            confidence = probabilities[i][predicted_class].item()
            
            results.append({
                'request_id': request_id,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
        
        # Clear pending requests
        self.pending_requests = []
        
        return results

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.throughput = []
    
    def log_inference(self, inference_time, memory_used, batch_size):
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_used)
        self.throughput.append(batch_size / inference_time)
    
    def get_stats(self):
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'p95_inference_time': np.percentile(self.inference_times, 95),
            'avg_memory_usage': np.mean(self.memory_usage),
            'avg_throughput': np.mean(self.throughput),
            'total_requests': len(self.inference_times)
        }

# Example deployment workflow
def deployment_workflow():
    # 1. Load trained model
    model = CNN(num_classes=10)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 2. Create example input
    example_input = torch.randn(1, 3, 224, 224)
    
    # 3. Optimize model
    optimized_model = optimize_model_for_inference(model, example_input)
    
    # 4. Quantize model (optional)
    # quantized_model = quantize_model(model, val_loader)
    
    # 5. Export to ONNX
    export_to_onnx(optimized_model, example_input, 'model.onnx')
    
    # 6. Optimize with TensorRT (optional)
    # optimize_with_tensorrt('model.onnx', 'model.trt')
    
    # 7. Test inference
    test_inference_performance(optimized_model, example_input)
    
    print("Deployment workflow completed!")

def test_inference_performance(model, example_input, num_runs=100):
    """Test inference performance"""
    model.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(example_input)
    
    # Measure performance
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(example_input)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = 1 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} inferences/second")

# Run deployment workflow
deployment_workflow()

if __name__ == '__main__':
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
```

## Conclusion

This comprehensive PyTorch tutorial covers the essential aspects of machine learning and deep learning implementation, from basic tensor operations to production deployment. Key takeaways include:

### Core Concepts Covered
1. **PyTorch Fundamentals**: Tensors, autograd, and basic operations
2. **Data Handling**: Custom datasets, data loaders, and preprocessing
3. **Neural Networks**: From simple MLPs to advanced architectures
4. **Training**: Optimization, loss functions, and training loops
5. **Computer Vision**: CNNs, transfer learning, and specialized architectures
6. **NLP**: RNNs, Transformers, and text processing
7. **Advanced Topics**: Custom losses, regularization, and interpretability
8. **Production**: Model optimization, quantization, and deployment

### Best Practices
- Use appropriate data augmentation and preprocessing
- Implement proper validation and early stopping
- Monitor training with comprehensive metrics
- Apply regularization techniques to prevent overfitting
- Optimize models for production deployment
- Use mixed precision training for efficiency
- Implement proper error handling and logging

### Next Steps
- Explore domain-specific applications
- Implement state-of-the-art architectures
- Experiment with distributed training
- Learn about model compression techniques
- Practice with real-world datasets
- Contribute to open-source projects

This tutorial provides a solid foundation for building and deploying machine learning models with PyTorch. Continue practicing with different datasets and architectures to master these concepts.

---

*Last updated: September 2025*