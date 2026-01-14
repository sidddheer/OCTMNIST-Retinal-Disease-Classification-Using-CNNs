import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

# --- Configuration ---
DATA_FLAG = 'octmnist'
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20
DROPOUT_RATE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders():
    # Preprocessing and Data Augmentation
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Download and Load OCTMNIST Data
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, info

class OCT_CNN(nn.Module):
    def __init__(self, num_classes):
        super(OCT_CNN, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # Fully Connected Layers
        # Input size is 28x28 -> after 3 pools (28->14->7->3) -> 3x3 spatial size
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)       # Dropout for regularization
        x = self.fc2(x)
        return x

def train():
    train_loader, val_loader, test_loader, info = get_data_loaders()
    num_classes = len(info['label'])
    
    model = OCT_CNN(num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Weighted Loss to handle Class Imbalance
    # (Weights should be calculated based on inverse class frequency in training set)
    # Example weights provided below (adjust based on your EDA)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 3.0]).to(DEVICE) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.squeeze().long().to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'octmnist_cnn.pth')
    print("Model saved.")

if __name__ == "__main__":
    train()
