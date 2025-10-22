#!/usr/bin/env python3
import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np
from pathlib import Path

LABELS = ['LEVEL_2','LEVEL_3','LEVEL_4','LEVEL_5','LEVEL_6','LEVEL_7']

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = timm.create_model("convnext_base", pretrained=False, num_classes=len(LABELS))
    # <- set the checkpoint filename you trained (place the file in project root)
    ckpt_path = Path("convnext_base_fold2_best.pth")
    sd = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(sd)
    except Exception:
        model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

_model = load_model()  # load once at import

# Preprocessing: make sure this matches training preprocessing
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

def predict(image: Image.Image):
    """
    Synchronous prediction function.
    Input: PIL.Image (RGB)
    Returns: (label:str, confidence:float)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = _preprocess(image).unsqueeze(0).to(device)  # 1xCxHxW
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])