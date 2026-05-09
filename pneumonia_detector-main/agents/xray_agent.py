import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features,2)

model.load_state_dict(
    torch.load("models/xray_model.pt",map_location=device)
)


model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

def analyze_xray_image(image_path: str) -> dict:
  
   
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabities = torch.softmax(outputs,dim=1)
        confidence, predicted_class = torch.max(probabities, 1)

    label_map = {0: "Normal",1: "Pneumonia"}

   
    result = {
        "agent": "X-ray Agent",
        "prediction": label_map[predicted_class.item()],
        "confidence": round(confidence.item(),2)
    }

    return result
