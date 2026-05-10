import io
import math
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from typing import List
import asyncio

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
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

# EMNIST Balanced label mapping (47 classes)
EMNIST_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

# Common dyslexia reversal pairs
REVERSAL_PAIRS = {
    'b': 'd', 'd': 'b',
    'p': 'q', 'q': 'p',
    'm': 'w', 'w': 'm',
    'n': 'u', 'u': 'n',
    '6': '9', '9': '6',
}

# ============================================================
# Auto-download models from Google Drive (for Railway deploy)
# ============================================================
import os

GDRIVE_FOLDER_ID = "1QllOzlbuSA3yXqvHPPhR8YfRX-xLiTVX"

MODEL_FILES = {
    "dyslexia_model_full.pt": None,
    "char_classifier.pt": None,
}

def download_models_if_needed():
    """Download model files from Google Drive if they don't exist locally."""
    missing = [f for f in MODEL_FILES if not os.path.exists(f)]
    if not missing:
        print("[OK] All model files found locally")
        return

    print(f"[INFO] Missing models: {missing}")
    print(f"[INFO] Downloading from Google Drive folder...")
    try:
        import gdown
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID,
            output=".",
            quiet=False,
            use_cookies=False,
        )
        print("[OK] Download complete")
    except Exception as e:
        print(f"[ERROR] Failed to download models: {e}")

download_models_if_needed()

# ============================================================
# Load Models
# ============================================================

# 1. Dyslexia quality model (existing)
dyslexia_model = None
try:
    dyslexia_model = torch.jit.load("dyslexia_model_full.pt", map_location=DEVICE)
    dyslexia_model.eval()
    print("[OK] Dyslexia quality model loaded")
except Exception as e:
    print(f"[WARN] Dyslexia quality model not found: {e}")

# 2. Character recognition model (new)
char_model = None
try:
    char_model = torch.jit.load("char_classifier.pt", map_location=DEVICE)
    char_model.eval()
    print("[OK] Character recognition model loaded")
except Exception as e:
    print(f"[WARN] Character recognition model not found: {e}")
    print("   (Run train_char_classifier.py to create it)")

# ============================================================
# Pre-build transforms (avoid recreating every call)
# ============================================================
_tf_dyslexia = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

_tf_char = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# ============================================================
# Shared Preprocessing (decode image ONCE)
# ============================================================

def _load_and_prepare_gray(image_bytes: bytes) -> Image.Image:
    """Load image bytes into a square, white-bg, grayscale PIL image. Done ONCE."""
    img_raw = Image.open(io.BytesIO(image_bytes))

    # Handle transparency
    if img_raw.mode in ('RGBA', 'LA') or (img_raw.mode == 'P' and 'transparency' in img_raw.info):
        bg = Image.new("RGB", img_raw.size, (255, 255, 255))
        bg.paste(img_raw, mask=img_raw.convert('RGBA').split()[3])
        img_gray = bg.convert("L")
    else:
        img_gray = img_raw.convert("L")

    # Pad to square
    w, h = img_gray.size
    if w != h:
        size = max(w, h)
        result = Image.new("L", (size, size), 255)
        result.paste(img_gray, ((size - w) // 2, (size - h) // 2))
        img_gray = result

    return img_gray


def _gray_to_dyslexia_tensor(img_gray: Image.Image) -> torch.Tensor:
    """Convert prepared grayscale image to dyslexia model tensor."""
    return _tf_dyslexia(img_gray.convert("RGB")).unsqueeze(0).to(DEVICE)


def _gray_to_char_tensor(img_gray: Image.Image) -> torch.Tensor:
    """Convert prepared grayscale image to char model tensor (inverted)."""
    img_inv = Image.eval(img_gray, lambda x: 255 - x)
    return _tf_char(img_inv).unsqueeze(0).to(DEVICE)


# ============================================================
# Character Recognition
# ============================================================

def recognize_char_from_tensor(tensor: torch.Tensor) -> tuple:
    """Recognize what character was written from pre-built tensor. Returns top5."""
    if char_model is None:
        return None, 0.0, []

    with torch.no_grad():
        output = char_model(tensor)
        probs = torch.softmax(output, dim=1)[0]

        top5_probs, top5_indices = torch.topk(probs, min(5, len(probs)))
        top5 = [(EMNIST_LABELS[idx.item()], prob.item()) for idx, prob in zip(top5_indices, top5_probs)]

        best_idx = top5_indices[0].item()
        best_char = EMNIST_LABELS[best_idx]
        best_conf = top5_probs[0].item()

    return best_char, best_conf, top5


def check_reversal(target: str, recognized: str) -> bool:
    """Check if the recognized char is a known dyslexia reversal of the target."""
    if not target or not recognized:
        return False
    t = target.lower()
    r = recognized.lower()
    return REVERSAL_PAIRS.get(t) == r


def normalize_char_for_comparison(char: str) -> str:
    """Normalize character for comparison (handle EMNIST merged classes)."""
    # EMNIST Balanced merges some upper/lowercase that look the same
    merged_to_lower = {
        'C': 'c', 'I': 'i', 'J': 'j', 'K': 'k', 'L': 'l',
        'M': 'm', 'O': 'o', 'P': 'p', 'S': 's', 'U': 'u',
        'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y', 'Z': 'z',
    }
    if char in merged_to_lower:
        return merged_to_lower[char]
    return char.lower()


def _find_char_in_top_predictions(target_lower: str, top_predictions: list) -> float:
    """Check if a character appears in the top predictions, return its confidence or 0."""
    for char, conf in top_predictions:
        if normalize_char_for_comparison(char) == target_lower:
            return conf
    return 0.0


# Minimum confidence to trust a "match" without deeper checking
CONFIDENCE_THRESHOLD = 0.50

def _analyze_single(image_bytes: bytes, target_char: str = None) -> dict:
    """Core analysis logic for a single image. Shared between /predict and /predict_batch."""
    # Step 0: Load and prepare image ONCE
    img_gray = _load_and_prepare_gray(image_bytes)

    # Step 1: Build tensors from shared grayscale (fast)
    char_tensor = _gray_to_char_tensor(img_gray) if char_model else None
    dyslexia_tensor = _gray_to_dyslexia_tensor(img_gray) if dyslexia_model else None

    # Step 2: Character Recognition (top5)
    recognized_char, char_confidence, top5 = (None, 0.0, [])
    if char_tensor is not None:
        recognized_char, char_confidence, top5 = recognize_char_from_tensor(char_tensor)
    recognized_normalized = normalize_char_for_comparison(recognized_char) if recognized_char else None

    # Step 3: Smart comparison with confidence cross-checking
    is_reversal = False
    is_mismatch = False
    match_method = "none"

    if target_char and recognized_char:
        target_normalized = target_char.lower()
        
        # Check: does the top-1 prediction match the target?
        top1_matches_target = target_normalized == (recognized_normalized or recognized_char.lower())

        # Check: is the top-1 prediction a reversal of the target?
        is_reversal = check_reversal(target_char, recognized_normalized or recognized_char)

        # Check for potential "confusers" in top-5
        # If user draws 'w', 'W' or 'V' might be in top-5
        confuser_chars = ['w', 'v', 'm', 'n'] if target_normalized in ['b', 'd'] else []
        highest_confuser_conf = 0.0
        for c in confuser_chars:
            conf = _find_char_in_top_predictions(c, top5)
            if conf > highest_confuser_conf:
                highest_confuser_conf = conf

        if top1_matches_target:
            if char_confidence >= CONFIDENCE_THRESHOLD:
                # Even if confident, if a strong confuser is present, be wary
                if highest_confuser_conf > char_confidence * 0.6:
                    is_mismatch = True
                    match_method = "confuser_detected_high_conf"
                else:
                    is_mismatch = False
                    match_method = "confident_match"
            else:
                # Model says "match" but with low confidence
                reversal_char = REVERSAL_PAIRS.get(target_normalized, None)
                reversal_in_top5_conf = _find_char_in_top_predictions(reversal_char, top5) if reversal_char else 0.0
                
                if reversal_in_top5_conf > 0 and reversal_in_top5_conf >= char_confidence * 0.6:
                    is_reversal = True
                    is_mismatch = True
                    match_method = "low_conf_reversal_detected"
                elif highest_confuser_conf > char_confidence * 0.5:
                    is_mismatch = True
                    match_method = "low_conf_confuser_detected"
                else:
                    is_mismatch = False
                    match_method = "low_conf_match"
        else:
            # Top-1 does NOT match target
            is_mismatch = True
            if is_reversal:
                match_method = "reversal"
            else:
                match_method = "clear_mismatch"

    # Step 4: Dyslexia quality model
    quality_probability = 0.5
    if dyslexia_tensor is not None:
        with torch.no_grad():
            output = dyslexia_model(dyslexia_tensor)
            logit = output.item() if output.numel() == 1 else output[0].item()
            quality_probability = 1.0 / (1.0 + math.exp(-logit))

    # Step 5: Combined Scoring
    indicators = []
    if char_model is not None and target_char:
        if match_method == "confuser_detected_high_conf" or match_method == "low_conf_confuser_detected":
            final_probability = max(0.6, quality_probability)
            indicators.append(
                f"Huruf diragukan: Model mengira '{recognized_char}' tapi terdeteksi kemiripan dengan huruf lain."
            )
        elif is_reversal:
            final_probability = max(0.85, quality_probability)
            indicators.append(
                f"Pembalikan huruf terdeteksi: diminta '{target_char}' tapi terbaca sebagai '{recognized_char}'"
            )
        elif is_mismatch:
            final_probability = max(0.65, quality_probability)
            indicators.append(
                f"Huruf tidak sesuai: diminta '{target_char}' tapi menulis '{recognized_char}'"
            )
        else:
            final_probability = quality_probability
            indicators.append(f"Huruf sesuai target: '{recognized_char}'")
    else:
        final_probability = quality_probability

    if quality_probability > 0.6:
        indicators.append("Pola pembentukan huruf tidak teratur terdeteksi")
    if quality_probability > 0.8:
        indicators.append("Goresan tidak konsisten (ketebalan dan arah)")

    is_dyslexic = final_probability > 0.5
    label = "DYSLEXIC" if is_dyslexic else "NON_DYSLEXIC"

    return {
        "prediction": label,
        "probability": float(final_probability),
        "is_dyslexic": is_dyslexic,
        "recognized_char": recognized_char,
        "char_confidence": float(char_confidence) if recognized_char else None,
        "target_char": target_char,
        "is_reversal": is_reversal,
        "is_mismatch": is_mismatch,
        "match_method": match_method,
        "top3_chars": [{"char": c, "confidence": round(p, 3)} for c, p in top5[:3]],
        "top5_chars": [{"char": c, "confidence": round(p, 3)} for c, p in top5],
        "quality_score": float(quality_probability),
        "indicators": indicators,
        "message": "Success",
    }


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def read_root():
    return {
        "message": "Grinbuds Dyslexia Detection API is running!",
        "dyslexia_model_loaded": dyslexia_model is not None,
        "char_model_loaded": char_model is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    target_char: str = Form(default=None),
):
    """
    Predict dyslexia indicators from a single handwriting image.
    """
    if dyslexia_model is None and char_model is None:
        raise HTTPException(status_code=500, detail="No models loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        return _analyze_single(image_bytes, target_char)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    target_chars: str = Form(default=""),
):
    """
    Predict dyslexia indicators for multiple handwriting images at once.
    Accepts up to 10 images. target_chars is a comma-separated string of target chars.
    Returns a list of predictions.
    
    This is ~3-5x faster than calling /predict multiple times because:
    - Single HTTP round-trip instead of N
    - Shared model warm state
    """
    if dyslexia_model is None and char_model is None:
        raise HTTPException(status_code=500, detail="No models loaded.")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch.")

    targets = [t.strip() for t in target_chars.split(",")] if target_chars else []

    results = []
    for i, file in enumerate(files):
        if not file.content_type.startswith("image/"):
            results.append({"error": f"File {i} is not an image", "index": i})
            continue

        image_bytes = await file.read()
        target = targets[i] if i < len(targets) else None

        try:
            result = _analyze_single(image_bytes, target)
            result["index"] = i
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "index": i})

    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
