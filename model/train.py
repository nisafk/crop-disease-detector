"""
train.py - This script trains our Crop Disease Detection AI model.

HOW IT WORKS:
1. We load the PlantVillage dataset (54,000 leaf images, 38 classes)
2. We take ResNet50 - a CNN already trained on 14 million general images
3. We replace its last layer with our own (38 disease classes instead of 1000)
4. We train it on our leaf images for a few epochs
5. We save the best model weights to a file called best_model.pth

RUN THIS ON GOOGLE COLAB (free GPU) - not on your local machine.
Instructions to run on Colab are at the bottom of this file.
"""

import torch                          # Main deep learning library
import torch.nn as nn                 # Neural network layers
import torch.optim as optim           # Optimizers (how model learns)
from torchvision import datasets, models, transforms  # Image tools + pretrained models
from torch.utils.data import DataLoader  # Loads batches of images efficiently
import os
import time
import copy

# ============================================================
# STEP 1: CONFIGURATION
# All settings in one place - easy to change
# ============================================================

DATA_DIR = "./plantvillage"       # Folder where your dataset is stored
MODEL_SAVE_PATH = "./best_model.pth"  # Where to save the trained model
NUM_CLASSES = 38                  # PlantVillage has 38 disease categories
BATCH_SIZE = 32                   # How many images to process at once
                                  # (Higher = faster but needs more RAM)
NUM_EPOCHS = 15                   # How many times to go through the full dataset
LEARNING_RATE = 0.001             # How fast the model learns
                                  # (Too high = overshoots, too low = slow)
IMAGE_SIZE = 224                  # ResNet50 expects 224x224 pixel images

# Use GPU if available (CUDA), otherwise CPU
# GPU is ~100x faster for training deep learning models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")  # Will print "cuda" on Colab GPU

# ============================================================
# STEP 2: DATA TRANSFORMS (IMAGE PREPROCESSING)
# Before feeding images to the model, we need to prepare them
# ============================================================

# Training transforms: we ADD random changes to make model more robust
# This is called "Data Augmentation" - artificially increases dataset variety
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),  # Random crop then resize to 224x224
    transforms.RandomHorizontalFlip(),          # Randomly flip image left-right
    transforms.RandomRotation(15),              # Randomly rotate up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary brightness/contrast
    transforms.ToTensor(),                      # Convert image to PyTorch tensor
                                                # (Changes pixels from 0-255 to 0.0-1.0)
    transforms.Normalize(                       # Normalize using ImageNet's mean/std
        mean=[0.485, 0.456, 0.406],            # These specific numbers come from
        std=[0.229, 0.224, 0.225]              # ImageNet dataset statistics
    )                                           # ResNet was trained with these, so we match
])

# Validation transforms: NO random changes - we want consistent evaluation
val_transforms = transforms.Compose([
    transforms.Resize(256),                    # Resize to slightly larger first
    transforms.CenterCrop(IMAGE_SIZE),         # Then crop center 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# STEP 3: LOAD THE DATASET
# ImageFolder automatically reads folder names as class labels
# Dataset structure:
#   plantvillage/
#     train/
#       Apple___Apple_scab/  <- folder name = class name
#         img1.jpg
#         img2.jpg
#       Apple___healthy/
#         ...
#     val/
#       ...
# ============================================================

print("Loading dataset...")
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'train'),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'val'),
    transform=val_transforms
)

# DataLoader: efficiently loads images in batches (not one by one)
# num_workers=4 means 4 parallel processes load images (faster on Colab)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Print class information
class_names = train_dataset.classes  # List of all 38 disease names
print(f"Number of classes: {len(class_names)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Classes: {class_names[:5]}...")  # Show first 5

# ============================================================
# STEP 4: BUILD THE MODEL (Transfer Learning)
# We use ResNet50 pretrained on ImageNet
# ResNet50 = 50-layer deep Residual Neural Network
# ============================================================

print("\nBuilding model with Transfer Learning...")

# Load ResNet50 with pretrained ImageNet weights
# weights=ResNet50_Weights.DEFAULT means use the best available pretrained weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# FREEZE all existing layers - we don't want to change what ResNet already learned
# (It already knows edges, textures, shapes from ImageNet)
for param in model.parameters():
    param.requires_grad = False   # requires_grad=False means "don't update this"

# REPLACE the final classification layer (originally 1000 classes for ImageNet)
# with our own for 38 disease classes
# model.fc = the "Fully Connected" final layer
# nn.Linear(2048, NUM_CLASSES) = linear layer: 2048 inputs → 38 outputs
model.fc = nn.Sequential(
    nn.Dropout(0.5),              # Dropout: randomly disable 50% neurons during training
                                   # This prevents overfitting (memorizing instead of learning)
    nn.Linear(2048, NUM_CLASSES)  # Our custom output layer: 2048 → 38
)

# Move model to GPU for faster training
model = model.to(DEVICE)

print("Model ready! Only the final layer will be trained.")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================
# STEP 5: DEFINE LOSS FUNCTION AND OPTIMIZER
# ============================================================

# Loss Function: CrossEntropyLoss
# Measures how wrong the model's predictions are
# If model says "95% Apple Scab" but actual is "Healthy" → high loss
# If model says "95% Apple Scab" and actual IS Apple Scab → low loss
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam
# Updates model weights after each batch to reduce loss
# Only optimize our new final layer (the rest is frozen)
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Learning Rate Scheduler
# Reduces learning rate by factor 0.1 every 7 epochs
# Like taking smaller steps as you get closer to the answer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ============================================================
# STEP 6: TRAINING LOOP
# This is where the model actually learns
# ============================================================

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    """
    Trains the model and returns the best version.
    
    Each epoch:
    1. Training phase: show model images, compute loss, update weights
    2. Validation phase: test on images model hasn't seen, measure accuracy
    3. Save model if it's the best accuracy so far
    """
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Each epoch has two phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Training mode: dropout is ACTIVE
                loader = train_loader
            else:
                model.eval()    # Evaluation mode: dropout is OFF
                loader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over batches of images
            for batch_idx, (inputs, labels) in enumerate(loader):
                # Move data to GPU
                inputs = inputs.to(DEVICE)   # Image tensors
                labels = labels.to(DEVICE)   # Correct class indices
                
                # Zero the parameter gradients
                # (Gradients accumulate by default, so we clear each batch)
                optimizer.zero_grad()
                
                # Forward pass: feed images through model, get predictions
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)          # Shape: (batch_size, 38)
                    _, preds = torch.max(outputs, 1) # Get class with highest score
                    loss = criterion(outputs, labels) # Calculate how wrong we are
                    
                    # Backward pass: only during training
                    if phase == 'train':
                        loss.backward()   # Calculate gradients (how to fix weights)
                        optimizer.step()  # Update weights
                
                # Track stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
            
            if phase == 'train':
                scheduler.step()  # Adjust learning rate
            
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            
            print(f"{phase.upper()} Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} ({epoch_acc*100:.2f}%)")
            
            # Save model if it's the best so far
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  ✅ New best model saved! Accuracy: {best_accuracy*100:.2f}%")
    
    print(f"\n✅ Training complete! Best validation accuracy: {best_accuracy*100:.2f}%")
    
    # Load best model weights before returning
    model.load_state_dict(best_model_weights)
    return model


# Also save the class names mapping (needed for prediction later)
import json
class_to_idx = train_dataset.class_to_idx  # {"Apple___Apple_scab": 0, ...}
idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse: {0: "Apple___Apple_scab", ...}

with open("class_names.json", "w") as f:
    json.dump(idx_to_class, f, indent=2)
print("Class names saved to class_names.json")

# ============================================================
# STEP 7: START TRAINING
# ============================================================
print("\nStarting training...")
start_time = time.time()

model = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)

elapsed = time.time() - start_time
print(f"\nTotal training time: {elapsed/60:.1f} minutes")
print(f"Model saved to: {MODEL_SAVE_PATH}")


# ============================================================
# HOW TO RUN THIS ON GOOGLE COLAB:
# ============================================================
# 1. Go to colab.research.google.com
# 2. New notebook → Runtime → Change runtime type → GPU (T4)
# 3. Run this first to get the dataset:
#
#    !pip install kaggle
#    from google.colab import files
#    files.upload()  # Upload your kaggle.json API key
#    !mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/
#    !kaggle datasets download -d abdallahalidev/plantvillage-dataset
#    !unzip plantvillage-dataset.zip -d plantvillage
#
# 4. Upload this train.py file to Colab
# 5. Run: !python train.py
# 6. After training, download best_model.pth and class_names.json
# ============================================================
