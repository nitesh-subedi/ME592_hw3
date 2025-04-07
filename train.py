import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import random_split
import torch.nn.functional as F
import torchvision.models as models

class DrivingDataset(Dataset):
    def _init_(self, csv_file):
        self.data = pd.read_csv(csv_file, delimiter=";")
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256)),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        
        # Fix path references if needed
        self.data["Path"] = self.data["Path"].apply(
            lambda x: x.replace("/home/nitesh/Downloads/Linux_Roversim/", "")
        )

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        row = self.data.iloc[idx]
        img_path = row['Path']
        steering_angle = row['SteerAngle']
        throttle = row['Throttle']

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
        # find mean and subtract mean
        mean = img_rgb.mean(axis=(0, 1))
        img_rgb = img_rgb - mean
        pil_img = T.functional.to_pil_image(img_rgb)

        tensor_img = self.transform(pil_img)
        label = torch.tensor([steering_angle, throttle], dtype=torch.float)

        return tensor_img, label
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
dataset = DrivingDataset(csv_file="train_data_2/robot_log.csv")
print(f"Dataset size: {len(dataset)}")
# Split the dataset into training and validation sets
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)


class ResNetSteeringThrottle(nn.Module):
    def _init_(self, backbone='resnet50', pretrained=True):
        super()._init_()
        
        # 1) Load a pretrained ResNet
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 2) Freeze all layers first (no grad for them)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # OPTIONAL: unfreeze the last residual block (layer4) to allow some fine-tuning
        # Comment out if you truly want to freeze everything
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # 3) Replace the final classification layer with Identity (pass-through).
        #    This means ResNet will output a feature vector of size ⁠ in_features ⁠.
        self.resnet.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(128, 2)  # [steering, throttle]
        )

    def forward(self, x):
        # Pass through ResNet to get features
        features = self.resnet(x)  # shape: (batch_size, in_features)
        # Pass through our new head
        out = self.head(features)  # shape: (batch_size, 2)
        return out

model = ResNetSteeringThrottle().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

num_epochs = 1000
# print(model.device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        # print(f"Images shape: {images.device}, Labels shape: {labels.device}")
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    val_loss = val_loss / len(val_loader.dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    # Save the model every 100 epochs
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f"train_data_2_models/model_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")
# Save the final model
torch.save(model.state_dict(), "train_data_2_models/model_final.pth")
print("Final model saved.")
# Save the model architecture
torch.save(model, "train_data_2_models/model_architecture.pth")
print("Model architecture saved.")