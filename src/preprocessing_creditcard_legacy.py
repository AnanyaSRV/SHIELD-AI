print("SCRIPT STARTED")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 50)
print("LOADING CREDIT CARD DATASET")
print("=" * 50)

df = pd.read_csv(r"D:\Ananya Mini\Datasets\creditcard.csv")

print("Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["Class"].value_counts())

X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Time and Amount
scaler = StandardScaler()

X["Time"] = scaler.fit_transform(X[["Time"]])
X["Amount"] = scaler.fit_transform(X[["Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

os.makedirs("data", exist_ok=True)

joblib.dump(X_train, "data/X_train.pkl")
joblib.dump(X_test, "data/X_test.pkl")
joblib.dump(y_train, "data/y_train.pkl")
joblib.dump(y_test, "data/y_test.pkl")
joblib.dump(scaler, "data/scaler.pkl")

print("\nPreprocessed data saved successfully.")