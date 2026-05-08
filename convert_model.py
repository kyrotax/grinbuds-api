"""
Convert best_model.pth (PyTorch weights) → dyslexia_model_full.pt (TorchScript)
Run this script from the backend_api folder.
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

class DyslexiaClassifier(nn.Module):
    """Same architecture as in train.py"""
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)  # No pretrained needed
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

def convert():
    print("Loading weights from best_model.pth...")
    model = DyslexiaClassifier()
    model.load_state_dict(
        torch.load("best_model.pth", map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print("[OK] Weights loaded successfully")

    print("Converting to TorchScript...")
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    scripted = torch.jit.trace(model, dummy)
    scripted.save("dyslexia_model_full.pt")
    
    size_mb = Path("dyslexia_model_full.pt").stat().st_size / 1024 / 1024
    print(f"[OK] Saved dyslexia_model_full.pt ({size_mb:.1f} MB)")
    print("Done! You can now restart the backend API.")

if __name__ == "__main__":
    convert()
