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
    
    # 3. Check polarity (white-on-black vs black-on-white)
    stat = img_with_bg.resize((50, 50))
    avg_brightness = sum(stat.getdata()) / (50 * 50)
    
    # 4. Invert if handwriting is white on dark background
    if avg_brightness < 128:
        img_final = Image.eval(img_with_bg, lambda x: 255 - x)
    else:
        img_final = img_with_bg
        
    img_final = img_final.convert("RGB")
    
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
