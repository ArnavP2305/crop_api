from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Crop Recommendation API")

# Input schema
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def home():
    return {
        "message": "Welcome to Crop Recommendation API 🌱",
        "usage": "Send a POST request to /predict with crop parameters in JSON."
    }

@app.post("/predict")
def predict_crop(input: CropInput):
    data = [[
        input.N, input.P, input.K, input.temperature,
        input.humidity, input.ph, input.rainfall
    ]]

    # Get probabilities
    probs = model.predict_proba(data)[0]

    # Map crops to probabilities
    crop_probs = dict(zip(model.classes_, probs))

    # Sort and get Top 3
    top_crops = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "recommended_crops": [
            {"crop": crop, "probability": round(prob*100, 2)}
            for crop, prob in top_crops
        ]
    }
