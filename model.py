import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

# CNN model definition (same as provided earlier)
class CNNFull(nn.Module):
    def __init__(self, n_classes):
        super(CNNFull, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1_input_size = 128 * 28 * 28  # Adjust for 224x224 input images
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.sigmoid(x)

# Custom dataset loader for images from two folders
class BikeDataset(Dataset):
    def __init__(self, same_bike_dir, different_bike_dir, transform=None):
        self.same_bike_images = [(os.path.join(same_bike_dir, img), 1) for img in os.listdir(same_bike_dir)]
        self.different_bike_images = [(os.path.join(different_bike_dir, img), 0) for img in os.listdir(different_bike_dir)]
        self.images = self.same_bike_images + self.different_bike_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize((224, 224))
            img = np.array(img).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img)

        return img, torch.tensor(label, dtype=torch.float32)

# Function to train the CNN and save checkpoint
def train_and_save_model(train_loader, model, checkpoint_path='cnn_checkpoint.pth', num_epochs=10):
    optimizer = optim.RMSprop(model.parameters())
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    print("Training complete. Saving checkpoint...")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")

# Main code to train CNN and save checkpoint
if __name__ == "__main__":
    same_bike_folder = "simmilar_image"
    different_bike_folder = "differant_image"

    train_dataset = BikeDataset(same_bike_folder, different_bike_folder)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the CNN model
    model = CNNFull(n_classes=1)

    # Train the model and save checkpoint
    train_and_save_model(train_loader, model, checkpoint_path='cnn_checkpoint.pth', num_epochs=10)