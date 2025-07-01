import marimo

__generated_with = "0.8.14"
app = marimo.App(width="medium")


@app.cell
def __():
    """
    # Computer Vision Model Training with NVIDIA GPUs
    
    This marimo notebook demonstrates training a computer vision model using NVIDIA GPU acceleration. 
    It showcases various GPU-intensive operations for testing the notebook analyzer.
    
    **Target Audience:** Computer Vision Engineers and ML Researchers
    
    **Estimated Time:** 1-3 hours depending on GPU configuration
    
    **NVIDIA Tools Used:** CUDA, cuDNN, Tensor Cores for mixed precision training
    """
    return


@app.cell
def __():
    # Environment setup and imports
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    from torchvision import models
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.cuda.amp import autocast, GradScaler
    import os
    from typing import Tuple, List
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    return (
        DataLoader,
        F,
        GradScaler,
        List,
        Tuple,
        autocast,
        models,
        np,
        nn,
        optim,
        os,
        plt,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def __(torch):
    # Configuration for high-performance training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 1000  # ImageNet classes
    BATCH_SIZE = 64  # Large batch size for GPU utilization
    IMAGE_SIZE = 224
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Mixed precision training settings
    USE_AMP = True  # Automatic Mixed Precision
    
    # Multi-GPU settings
    USE_MULTI_GPU = torch.cuda.device_count() > 1
    
    print(f"Training configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Mixed precision: {USE_AMP}")
    print(f"  Multi-GPU: {USE_MULTI_GPU}")
    
    return (
        BATCH_SIZE,
        DEVICE,
        IMAGE_SIZE,
        LEARNING_RATE,
        NUM_CLASSES,
        NUM_EPOCHS,
        USE_AMP,
        USE_MULTI_GPU,
        WEIGHT_DECAY,
    )


@app.cell
def __(BATCH_SIZE, IMAGE_SIZE, transforms, torchvision):
    # Data loading and preprocessing with GPU-optimized transforms
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load ImageNet dataset (using CIFAR-10 as a substitute for demonstration)
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_val
    )
    
    # Data loaders with optimized settings for GPU training
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Multi-threaded data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    return train_dataset, train_loader, transform_train, transform_val, val_dataset, val_loader


@app.cell
def __(DEVICE, USE_MULTI_GPU, models, nn, torch):
    # Model architecture - Large ResNet with custom head
    class AdvancedResNet(nn.Module):
        def __init__(self, num_classes=1000, pretrained=True):
            super(AdvancedResNet, self).__init__()
            
            # Use ResNet-152 as backbone (very large model)
            self.backbone = models.resnet152(pretrained=pretrained)
            
            # Remove the final classification layer
            self.backbone.fc = nn.Identity()
            
            # Custom classification head with more parameters
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(1024, num_classes)
            )
            
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    # Initialize model
    model = AdvancedResNet(num_classes=10)  # CIFAR-10 has 10 classes
    
    # Move to GPU and setup multi-GPU if available
    model = model.to(DEVICE)
    if USE_MULTI_GPU:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for training")
    
    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: Advanced ResNet-152")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB (fp32)")
    
    return AdvancedResNet, model, total_params, trainable_params


@app.cell
def __(LEARNING_RATE, WEIGHT_DECAY, model, optim, torch):
    # Optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=100,  # Number of epochs
        eta_min=1e-6
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Mixed precision scaler for AMP
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print("Training setup complete:")
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Loss function: CrossEntropyLoss")
    print(f"  Mixed precision: {scaler is not None}")
    
    return criterion, optimizer, scaler, scheduler


@app.cell
def __(DEVICE, USE_AMP, autocast, criterion, model, optimizer, scaler, torch, train_loader):
    # Training loop with advanced GPU optimizations
    def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if USE_AMP and scaler is not None:
                # Mixed precision forward pass
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    print("Training function defined with optimizations:")
    print("  - Non-blocking GPU transfers")
    print("  - Efficient gradient zeroing")
    print("  - Mixed precision training")
    print("  - Memory-optimized operations")
    
    return (train_epoch,)


@app.cell
def __(torch):
    # Memory usage monitoring
    def print_gpu_memory():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                
                print(f"GPU {i} Memory:")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved: {reserved:.2f} GB")
                print(f"  Peak: {max_allocated:.2f} GB")
        else:
            print("CUDA not available")
    
    # Performance profiling
    print("GPU memory status before training:")
    print_gpu_memory()
    
    return (print_gpu_memory,)


@app.cell
def __():
    """
    ## Training Execution and Performance Analysis
    
    This section would execute the training loop with:
    
    1. **Large Model Training**: ResNet-152 with custom head (~60M parameters)
    2. **Multi-GPU Utilization**: DataParallel across available GPUs
    3. **Mixed Precision**: FP16 training for memory efficiency and speed
    4. **Advanced Optimizations**: Non-blocking transfers, gradient scaling, efficient memory usage
    
    **Expected GPU Requirements:**
    - Minimum: 1x RTX 3080 12GB or 1x RTX 4080 16GB  
    - Optimal: 2x RTX 4090 24GB or 1x A100 40GB
    - For batch_size=64: ~8-12GB VRAM per GPU
    
    **Key Features Detected:**
    - Computer vision workload (image classification)
    - Large batch sizes requiring significant VRAM
    - Mixed precision training (tensor cores beneficial)
    - Multi-GPU data parallelism
    - Memory-intensive model (ResNet-152)
    """
    return


@app.cell
def __(print_gpu_memory, torch):
    # Final performance summary
    print("🚀 Training Configuration Summary:")
    print("="*50)
    print("Model: Advanced ResNet-152 (60M+ parameters)")
    print("Workload: Computer Vision - Image Classification")
    print("Optimization: Mixed Precision + Multi-GPU")
    print("Memory: High VRAM usage with large batches")
    print("Hardware: Tensor Core acceleration recommended")
    print("="*50)
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 GPU memory cleared")
        print_gpu_memory()
    
    return


if __name__ == "__main__":
    app.run() 