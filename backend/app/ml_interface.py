import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np

LABELS = ['LEVEL_2','LEVEL_3','LEVEL_4','LEVEL_5','LEVEL_6','LEVEL_7']

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = timm.create_model("convnext_base", pretrained=False, num_classes=6)
    model.load_state_dict(torch.load("best_convnext_hairfall.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()  # Load once at startup

def predict(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    img_tensor = preprocess(image).unsqueeze(0).to(device)  # Move tensor to device
    with torch.no_grad():
        output = model(img_tensor)
        proba = torch.softmax(output, dim=1).cpu().numpy().squeeze()
    pred_idx = np.argmax(proba)
    return LABELS[pred_idx], float(proba[pred_idx])