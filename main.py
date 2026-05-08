import io
import math
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

app = FastAPI(title="Grinbuds Dyslexia Detection API")

# Enable CORS so web apps can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IMAGE_SIZE = 96
MODEL_PATH = "dyslexia_model_full.pt"
DEVICE = torch.device("cpu")

# Load model globally on startup
try:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    # 1. Open image
    img_raw = Image.open(io.BytesIO(image_bytes))
    
    # 2. Handle transparency (replace alpha with white background)
    if img_raw.mode in ('RGBA', 'LA') or (img_raw.mode == 'P' and 'transparency' in img_raw.info):
        bg = Image.new("RGB", img_raw.size, (255, 255, 255))
        bg.paste(img_raw, mask=img_raw.convert('RGBA').split()[3])
        img_with_bg = bg.convert("L")
    else:
        img_with_bg = img_raw.convert("L")
    
    # Helper function to pad image to square to prevent distortion
    def pad_to_square(img):
        width, height = img.size
        if width == height:
            return img
        size = max(width, height)
        result = Image.new(img.mode, (size, size), 255) # 255 is white background
        result.paste(img, ((size - width) // 2, (size - height) // 2))
        return result
        
    img_padded = pad_to_square(img_with_bg)
    
    # 3. Assume real-world photos are dark ink on light background.
    # Auto-invert is disabled because room shadows can trigger false inversions
    # and ruin the prediction.
    img_final = img_padded.convert("RGB")
    
    # 5. Transform for model
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return tf(img_final).unsqueeze(0).to(DEVICE)

@app.get("/")
def read_root():
    return {"message": "Grinbuds Dyslexia Detection API is running!", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            output = model(tensor)
            
            # Extract logit (output could be a scalar tensor or single element tensor)
            logit = output.item() if output.numel() == 1 else output[0].item()
            
            # Apply sigmoid
            probability = 1.0 / (1.0 + math.exp(-logit))
            
            is_dyslexic = probability > 0.5
            label = "DYSLEXIC" if is_dyslexic else "NON_DYSLEXIC"
            
            return {
                "prediction": label,
                "probability": float(probability),
                "is_dyslexic": is_dyslexic,
                "message": "Success"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
