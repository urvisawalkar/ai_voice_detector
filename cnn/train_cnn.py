import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DEVICE = "cpu"  # you have no GPU

# --------------------------------
# Image transforms
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("spectrograms", transform=transform)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --------------------------------
# Simple CNN (ResNet18 pretrained)
# --------------------------------
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------------
# Training
# --------------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "cnn_model.pth")
print("\nCNN model saved: cnn_model.pth")
