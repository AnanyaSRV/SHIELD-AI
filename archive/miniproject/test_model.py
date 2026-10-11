import joblib

model = joblib.load("models/fraud_ensemble.pkl")

print("Model type:")
print(type(model))

print("\nHas predict:")
print(hasattr(model, "predict"))

print("\nHas predict_proba:")
print(hasattr(model, "predict_proba"))

print("\nModel:")
print(model)