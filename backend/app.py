from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import os
import base64

app = FastAPI(title="Factory Inspection API")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_SIZE = 128

THRESHOLDS = {
    "bottle": 0.0025698007,
    "cable": 0.00556921,
    "capsule": 0.000610792078078,
    "screw": 0.001084416988306,
    "metal_nut": 0.002574885264039}

print("Loading models into memory...")
loaded_models = {}

for category in THRESHOLDS.keys():
    model_path = BASE_DIR / "model" / f"{category}_defect_autoencoder.keras"
    try:
        loaded_models[category] = tf.keras.models.load_model(model_path)
        print(f"Loaded {category} successfully.")
    except Exception as e:
        print(f"Failed to load {category}: {e}")

print("All available models loaded.")

def preprocess_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Could not read image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

@app.get("/")
def home():
    return {"message": "Multi-Category Factory Inspection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), category: str = Form(...)):
    category = category.lower().strip()

    if category not in THRESHOLDS or category not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Available: {list(THRESHOLDS.keys())}"
        )

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        # Preprocess image
        image = preprocess_image(temp_path)

        # Select correct model and threshold dynamically
        model = loaded_models[category]
        threshold = THRESHOLDS[category]

        # Reconstruct image
        reconstructed = model.predict(image, verbose=0)

        # Calculate reconstruction error
        diff = np.abs(image[0] - reconstructed[0])
        error = np.mean(np.square(diff))

        # Classify
        prediction = "Defective" if error > threshold else "Good"

        # --- GENERATE HEATMAP ---
        # 1. Convert the difference array to a visual 0-255 scale
        diff_norm = (diff / diff.max()) * 255.0
        diff_norm = diff_norm.astype(np.uint8)

        # 2. Convert to grayscale
        gray_diff = cv2.cvtColor(diff_norm, cv2.COLOR_RGB2GRAY)

        # 3. Apply a thermal colormap (JET) to highlight high differences in red
        heatmap = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)

        # 4. Encode as Base64 string to send to the browser
        _, buffer = cv2.imencode('.jpg', heatmap)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "category": category,
            "prediction": prediction,
            "reconstruction_error": float(error),
            "threshold": threshold,
            "heatmap": heatmap_b64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)