import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

train_dataset = datasets.ImageFolder(
    root="data/chest_xray/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root = "data/chest_xray/val",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)

print("Classes:", train_dataset.classes)
print("Number of training images:",len(train_dataset))
print("Number of Validation images:",len(val_dataset))


model = models.resnet18(pretrained=True)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features,2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Using device:", device)
print("ResNet18 model created and ready for training")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

num_epochs = 3 

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Training loss: {avg_loss:.4f}")

model.eval()
correct=0
total=0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total if total > 0 else 0
print(f"Validation Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "models/xray_model.pt")
print("Model saved as models/xray_model.pt")