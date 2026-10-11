# backend.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = "models/rf_frontend.pkl"

print("Loading frontend model...")

model = joblib.load(MODEL_PATH)

print("Frontend model loaded successfully.")

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="UPI Fraud Detection API",
    description="Java Swing + Random Forest Fraud Detection",
    version="1.0"
)


# ============================================
# REQUEST SCHEMA
# ============================================

class TransactionRequest(BaseModel):
    senderCard: int
    senderAddr: int
    receiverCard: int
    receiverAddr: int
    amount: float


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")
def home():
    return {
        "message": "UPI Fraud Detection Backend Running"
    }


# ============================================
# PREDICTION ENDPOINT
# ============================================

@app.post("/predict")
def predict(transaction: TransactionRequest):

    try:

        # Use only features used during training
        features = pd.DataFrame(
            [[
                transaction.senderCard,
                transaction.senderAddr,
                transaction.amount
            ]],
            columns=[
                "card1",
                "addr1",
                "TransactionAmt"
            ]
        )

        prediction = model.predict(features)[0]

        probability = float(
            model.predict_proba(features)[0][1]
        )

        if prediction == 1:
            label = "Fraud"
        else:
            label = "Legitimate"

        return {
            "prediction": label,
            "fraud_probability": round(probability, 4)
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )