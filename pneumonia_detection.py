import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


base_path = 'data/chest_xray_dataset'

# Define paths for each split
train_dir = os.path.join(base_path, 'train')
val_dir = os.path.join(base_path, 'val')
test_dir = os.path.join(base_path, 'test')

normal_train_dir = os.path.join(train_dir, 'NORMAL')
pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')
normal_test_dir = os.path.join(test_dir, 'NORMAL')
pneumonia_test_dir = os.path.join(test_dir, 'PNEUMONIA')
normal_val_dir = os.path.join(val_dir, 'NORMAL')
pneumonia_val_dir = os.path.join(val_dir, 'PNEUMONIA')


# Create custom dataset class to load images
class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label




input_size = 224

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training transforms WITH augmentation
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),           # Rotate up to 10 degrees
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Zoom
    transforms.RandomHorizontalFlip(),       # Horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Validation transforms - NO augmentation, same as before
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# ## Create DataLoaders

train_dataset = ChestXrayDataset(
    data_dir = train_dir,
    transform = train_transforms
)

val_dataset = ChestXrayDataset(
    data_dir = val_dir,
    transform = val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# #### Train the model

def train_and_evaluate(model, optimizer, train_loader, val_loader, criterion, num_epochs, device):
    best_val_accuracy = 0.0  # Initialize variable to track the best validation accuracy
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over the training data
        for inputs, labels in train_loader:
            # Move data to the specified device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
    
            # Zero the parameter gradients to prevent accumulation
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(1)     # (B,)
            labels = labels.float()          # 0. or 1.
            # Calculate the loss
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            running_loss += loss.item()
            # Get predictions
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long()
            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
    
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Disable gradient calculation for validation
        with torch.no_grad():
            # Iterate over the validation data
            for inputs, labels in val_loader:
                # Move data to the specified device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                outputs = model(inputs)
                # Calculate the loss
                outputs = outputs.squeeze(1)     # (B,)
                labels = labels.float()          # 0. or 1.
                loss = criterion(outputs, labels)
    
                # Accumulate validation loss
                val_loss += loss.item()
                # Get predictions
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).long()
                # Update total and correct predictions
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
    
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Checkpoint the model if validation accuracy improved
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            dir = 'model'
            checkpoint = f'mobilenet_v2_{epoch+1:02d}_{val_acc:.3f}.pth'
            checkpoint_path = os.path.join(dir, checkpoint)
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaDetectionMobileNet(num_classes=1)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()


# ## MobileNetV2 Transfer Learning

class PneumoniaDetectionMobileNet(nn.Module):
    def __init__(self, droprate=0.2, num_classes=1):
        super(PneumoniaDetectionMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Remove original classifier
        self.base_model.classifier = nn.Identity()

        # Add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(droprate)  # Add dropout
        self.output_layer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout
        x = self.output_layer(x)
        return x


def make_model(learning_rate=0.001, droprate=0.2):
    model = PneumoniaDetectionMobileNet(num_classes=1, droprate=droprate)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


num_epochs = 3
model, optimizer = make_model(learning_rate=0.001, droprate=dr)
train_and_evaluate(model, optimizer, train_loader, val_loader, criterion, num_epochs, device)


dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = "model/pneumonia_mobilenet_v2.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)

print(f"Model exported to {onnx_path}")