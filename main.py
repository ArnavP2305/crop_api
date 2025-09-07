import numpy as np
from fastapi import FastAPI
import pickle

# Load trained model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    input_data = np.array([[data['N'], data['P'], data['K'],
                            data['temperature'], data['humidity'],
                            data['ph'], data['rainfall']]])
    
    # Get probabilities for all crops
    probs = model.predict_proba(input_data)[0]
    crop_classes = model.classes_
    
    
    top_n = 12  
    top_indices = np.argsort(probs)[::-1][:top_n]
    
    recommendations = [
        {"crop": crop_classes[i], "probability": round(probs[i], 4)}
        for i in top_indices
    ]
    
    return {"recommendations": recommendations}
