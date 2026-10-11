import joblib

model = joblib.load("models/fraud_ensemble.pkl")

print(type(model))
print(model)