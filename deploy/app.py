from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np

app = FastAPI(title="AutoOps AI Backend")

# 1. Load the Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_v1.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# 2. Define Input
class PredictionInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Convert input to DataFrame
        input_data = data.model_dump()
        df = pd.DataFrame([input_data])
        
        # Make Prediction (0 or 1)
        prediction = model.predict(df)[0]
        
        # === THE FIX: GET REAL PROBABILITY ===
        # predict_proba returns [[prob_0, prob_1]]
        # We take index 1 (Probability of Churn)
        probability = model.predict_proba(df)[0][1]
        
        # Extract Feature Importance
        classifier = model.named_steps['classifier']
        importances = classifier.feature_importances_.tolist()
        
        return {
            "prediction": int(prediction), 
            "probability": float(probability), # Sending real confidence score
            "importances": importances
        }
    
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))