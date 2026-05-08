# Grinbuds API - Dyslexia Detection 🧠

This is the backend API for **Grinbuds**, a Dyslexia Handwriting Detection App. This API serves a PyTorch machine learning model that analyzes images of handwriting and predicts the likelihood of dyslexia.

## 🚀 Tech Stack
- **Framework:** FastAPI (Python)
- **Machine Learning:** PyTorch, TorchVision (MobileNetV2)
- **Image Processing:** Pillow (PIL)
- **Deployment:** Railway / Docker

## 🛠️ Features
- Loads a pre-trained PyTorch TorchScript model (`dyslexia_model_full.pt`).
- Accepts image uploads via the `/predict` endpoint.
- Automatically handles image transparency and polarity (white-on-black vs black-on-white) to match the training dataset.
- Returns the prediction label, confidence probability, and a boolean flag.

---

## 📡 API Endpoints

### 1. Health Check
Checks if the API is running and if the ML model has been loaded successfully.
- **URL:** `/`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "message": "Grinbuds Dyslexia Detection API is running!",
    "model_loaded": true
  }
  ```

### 2. Predict Dyslexia
Analyzes an uploaded handwriting image.
- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:** 
  - `file`: (File) The image file to analyze (PNG, JPG, JPEG).
- **Response:**
  ```json
  {
    "prediction": "DYSLEXIC",
    "probability": 0.854,
    "is_dyslexic": true,
    "message": "Success"
  }
  ```

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kyrotax/grinbuds-api.git
   cd grinbuds-api
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Server:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Test the API:**
   Open your browser and go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to use the interactive Swagger UI and test the `/predict` endpoint easily.

---

## ☁️ Deployment (Railway)

This repository is configured to be deployed effortlessly on [Railway](https://railway.app). 
The `Procfile` is already included to tell Railway how to start the Uvicorn server, and `requirements.txt` is configured to use the CPU-only version of PyTorch to save memory.

1. Go to Railway and create a **New Project**.
2. Select **Deploy from GitHub repo**.
3. Select this repository.
4. Railway will automatically build and deploy the API!
