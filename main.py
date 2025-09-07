from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the trained model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Crop Recommendation API")

# Define input data format
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict")
def predict_crop(data: CropInput):
    # Convert input to list for model prediction
    features = [[
        data.N, data.P, data.K, 
        data.temperature, data.humidity, 
        data.ph, data.rainfall
    ]]
    prediction = model.predict(features)
    return {"recommended_crop": prediction[0]}

@app.get("/")
def home():
    return {"message": "Welcome to Crop Recommendation API. Use /predict endpoint."}

