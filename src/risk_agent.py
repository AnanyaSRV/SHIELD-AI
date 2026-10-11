
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "reports"
    / "lightgbm_tuning"
    / "best_validation_model.pkl"
)

PREPROCESSING_PATH = (
    ROOT
    / "reports"
    / "lightgbm_tuning"
    / "preprocessing.pkl"
)


class RiskAssessmentAgent:
    """Risk assessment using the saved LightGBM model and preprocessing schema."""

    def __init__(self, threshold=0.371294741468585):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        if not PREPROCESSING_PATH.is_file():
            raise FileNotFoundError(
                f"Preprocessing file not found: {PREPROCESSING_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.preprocessing = joblib.load(PREPROCESSING_PATH)

        self._load_preprocessing_metadata()

        # This is the previously selected validation threshold.
        self.threshold = float(threshold)

        self.explainer = None

    def _load_preprocessing_metadata(self):
        """Load feature names, encoder and numeric imputation values."""

        prep = self.preprocessing

        if not isinstance(prep, dict):
            raise TypeError(
                "Expected preprocessing.pkl to contain a dictionary. "
                "Adjust this method to match its saved structure."
            )

        self.feature_columns = prep.get("feature_columns")
        self.categorical_columns = prep.get("categorical_columns", [])
        self.numeric_columns = prep.get("numeric_columns", [])
        self.numeric_medians = prep.get("numeric_medians", {})
        self.encoder = prep.get("encoder")

        if not self.feature_columns:
            raise ValueError(
                "The preprocessing file does not contain feature_columns."
            )

        self.feature_columns = list(self.feature_columns)
        self.categorical_columns = list(self.categorical_columns)
        self.numeric_columns = list(self.numeric_columns)

        if self.encoder is None and self.categorical_columns:
            raise ValueError(
                "Categorical columns exist, but no saved encoder was found."
            )

    def _preprocess_batch(self, transaction_features):
        """Preprocess multiple transactions using the saved feature schema."""

        if not isinstance(transaction_features, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if transaction_features.empty:
            raise ValueError("Input batch contains no rows.")

        if "isFraud" in transaction_features.columns:
            raise ValueError(
                "Do not include the target column 'isFraud' in prediction input."
            )

        frame = transaction_features.drop(
            columns=["TransactionID"],
            errors="ignore",
        )

        missing = set(self.feature_columns) - set(frame.columns)
        extra = set(frame.columns) - set(self.feature_columns)

        if missing or extra:
            raise ValueError(
                "Feature schema mismatch. "
                f"Missing ({len(missing)}): {sorted(missing)[:10]}; "
                f"unexpected ({len(extra)}): {sorted(extra)[:10]}."
            )

        frame = frame[self.feature_columns].copy()

        for column in self.categorical_columns:
            frame[column] = (
                frame[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

        if self.categorical_columns:
            frame.loc[:, self.categorical_columns] = (
                self.encoder.transform(
                    frame[self.categorical_columns]
                )
            )

        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            median = self.numeric_medians.get(column, 0.0)
            frame[column] = frame[column].fillna(median)

        frame = frame[self.feature_columns].astype(np.float32)

        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(
                "Preprocessed batch contains infinite or invalid values."
            )

        return frame

    def _preprocess(self, transaction_features):
        """Preprocess one transaction while preserving the batch schema."""

        if isinstance(transaction_features, dict):
            transaction_features = pd.DataFrame(
                [transaction_features]
            )

        if not isinstance(transaction_features, pd.DataFrame):
            raise TypeError(
                "Input must be a dictionary or pandas DataFrame."
            )

        processed = self._preprocess_batch(transaction_features)

        if len(processed) != 1:
            raise ValueError(
                "_preprocess() accepts exactly one transaction. "
                "Use _preprocess_batch() for multiple transactions."
            )

        return processed

    def assess_risk(self, transaction_features):
        """Return a risk assessment for one transaction."""

        processed = self._preprocess(transaction_features)

        probability = float(
            self.model.predict_proba(processed)[0, 1]
        )

        prediction = int(probability >= self.threshold)

        return {
            "fraud_probability": probability,
            "risk_score": probability * 100.0,
            "prediction": prediction,
            "classification": (
                "Fraud" if prediction == 1 else "Legitimate"
            ),
            "classification_threshold": self.threshold,
        }
