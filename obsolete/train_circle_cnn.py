import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import re
import time
# Define the dataset class
import torch
import cv2
import numpy as np
import os
import re
from torch.utils.data import Dataset

class CircleDataset(Dataset):
    def __init__(self, image_folders, labels_files, transform=None):
        """
        Loads data from multiple folders.
        
        :param image_folders: List of directories containing images
        :param labels_files: List of corresponding label files
        :param transform: Optional transformation for images
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Ensure input lists match
        assert len(image_folders) == len(labels_files), "Mismatch between folders and labels"

        # Load all images and labels from multiple directories
        for folder, label_file in zip(image_folders, labels_files):
            folder_images = sorted(
                [f for f in os.listdir(folder) if re.fullmatch(r'\d+\.png', f)],
                key=lambda x: int(x.split('.')[0])
            )
            folder_labels = np.loadtxt(label_file, delimiter=',')

            # Ensure the number of images matches the number of labels
            assert len(folder_images) == len(folder_labels), f"Label mismatch in {folder}"

            # Store full paths and labels
            self.image_paths.extend([os.path.join(folder, img) for img in folder_images])
            self.labels.extend(folder_labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to 480x480
        image = cv2.resize(image, (480, 480))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 480, 480)
        image = image / 255.0  # Normalize

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label  # Image shape: (1, 480, 480), label shape: (2,)

class CircleDataset2(Dataset):
    def __init__(self, image_folder, labels_file, transform=None):
        self.image_folder = image_folder
        self.labels = np.loadtxt(labels_file, delimiter=',')
        self.transform = transform

        # Only select filenames that match a numeric pattern (e.g., 0.png, 1.png, ...)
        self.image_filenames = sorted(
            [f for f in os.listdir(image_folder) if re.fullmatch(r'\d+\.png', f)],
            key=lambda x: int(x.split('.')[0])
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__2(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to 480x480
        image = cv2.resize(image, (480, 480))
        image = np.expand_dims(image, axis=2)  # Add channel dimension (1, 480, 480)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)  # Ensures (1, 480, 480) shape
        return image, label  # No need to use torch.tensor() again
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to 480x480
        image = cv2.resize(image, (480, 480))

        # Convert to tensor and ensure correct shape
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 480, 480)

        # Normalize pixel values to [0,1]
        image = image / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label  # Correct shape: (1, 480, 480)

class CircleCNN(nn.Module):
    def __init__(self):
        super(CircleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Conv Layer 1: 7x7 kernel, 64 filters, stride 2
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 max pooling

            # Conv Layer 2: 3x3 kernel, 128 filters, stride 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 max pooling

            # Conv Layer 3: 3x3 kernel, 256 filters, stride 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling
        )

        # Compute the final flattened feature size
        dummy_input = torch.zeros(1, 1, 480, 480)  # Example input
        conv_output_size = self._get_conv_output_size(dummy_input)

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)  # Output x, y coordinates
        )

    def _get_conv_output_size(self, x):
        """Helper function to determine the flattened size of conv layers output"""
        x = self.conv_layers(x)
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
    def predict(self, image):
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 480, 480) 
        image_tensor = image_tensor / 255.0  # Normalize
        image_tensor = image_tensor.to('cuda')

        # Run inference
        with torch.no_grad():
            predicted_center = self.forward(image_tensor).cpu().numpy()[0]  # Get (x, y) prediction

        # Convert predicted coordinates to integer
        pred_x, pred_y = int(predicted_center[0]), int(predicted_center[1])
        return [pred_x, pred_y]

# Define the CNN model
class CircleCNN2(nn.Module):
    def __init__(self):
        super(CircleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * (480 // 8) * (480 // 8), 256),  # Adjust based on input size
            nn.ReLU(),
            nn.Linear(256, 2)  # Output x, y coordinates
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Training setup
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_folders = ["data/set-4", "data/set-5"]
    labels_files = ["data/set-4/new_center.txt", "data/set-5/new_center.txt"]

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = CircleDataset(image_folders, labels_files, transform)
    train_loader = DataLoader(dataset, batch_size=48, shuffle=True)

    # Initialize model, loss function, optimizer
    model = CircleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "circle_detector.pth")
    print("Training complete. Model saved as 'circle_detector.pth'.")
