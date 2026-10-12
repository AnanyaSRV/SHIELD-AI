
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

DEFAULT_THRESHOLD = 0.371294741468585
TOP_EXPLANATION_FEATURES = 5


class RiskAssessmentAgent:
    """Risk assessment using the saved LightGBM model and preprocessing schema."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
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

        self.threshold = float(threshold)

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "Classification threshold must be between 0 and 1."
            )

        self.explainer = None
        self.explanation_initialization_error = None

    def _load_preprocessing_metadata(self):
        """Load feature names, encoder and numeric imputation values."""

        prep = self.preprocessing

        if not isinstance(prep, dict):
            raise TypeError(
                "Expected preprocessing.pkl to contain a dictionary."
            )

        self.feature_columns = prep.get("feature_columns")
        self.categorical_columns = prep.get(
            "categorical_columns", []
        )
        self.numeric_columns = prep.get("numeric_columns", [])
        self.numeric_medians = prep.get("numeric_medians", {})
        self.encoder = prep.get("encoder")

        if self.feature_columns is None or len(self.feature_columns) == 0:
            raise ValueError(
                "The preprocessing file does not contain feature_columns."
            )

        self.feature_columns = list(self.feature_columns)
        self.categorical_columns = list(self.categorical_columns)
        self.numeric_columns = list(self.numeric_columns)

        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError(
                "The saved feature schema contains duplicate column names."
            )

        unknown_columns = (
            set(self.categorical_columns)
            | set(self.numeric_columns)
        ) - set(self.feature_columns)

        if unknown_columns:
            raise ValueError(
                "Preprocessing metadata contains unknown columns: "
                f"{sorted(unknown_columns)[:10]}"
            )

        if self.encoder is None and self.categorical_columns:
            raise ValueError(
                "Categorical columns exist, but no saved encoder was found."
            )

    def _preprocess_batch(self, transaction_features):
        """Preprocess transactions using the saved feature schema."""

        if not isinstance(transaction_features, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if transaction_features.empty:
            raise ValueError(
                "Input batch contains no rows."
            )

        if "isFraud" in transaction_features.columns:
            raise ValueError(
                "Do not include the target column 'isFraud' "
                "in prediction input."
            )

        frame = transaction_features.drop(
            columns=["TransactionID"],
            errors="ignore",
        )

        missing = (
            set(self.feature_columns) - set(frame.columns)
        )
        extra = (
            set(frame.columns) - set(self.feature_columns)
        )

        if missing or extra:
            raise ValueError(
                "Feature schema mismatch. "
                f"Missing ({len(missing)}): "
                f"{sorted(missing)[:10]}; "
                f"unexpected ({len(extra)}): "
                f"{sorted(extra)[:10]}."
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
            encoded = self.encoder.transform(
                frame[self.categorical_columns]
            )

            if hasattr(encoded, "toarray"):
                encoded = encoded.toarray()

            encoded = np.asarray(encoded)

            if encoded.shape != (
                len(frame),
                len(self.categorical_columns),
            ):
                raise ValueError(
                    "The saved encoder output shape does not match "
                    "the configured categorical columns. Check "
                    "preprocessing.pkl and the training pipeline."
                )

            frame.loc[:, self.categorical_columns] = encoded

        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            median = self.numeric_medians.get(column, 0.0)
            frame[column] = frame[column].fillna(median)

        frame = frame[self.feature_columns].astype(np.float32)

        values = frame.to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                "Preprocessed input contains infinite or invalid values."
            )

        return frame

    def _preprocess(self, transaction_features):
        """Preprocess exactly one transaction."""

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

    def _get_explainer(self):
        """Initialize the SHAP TreeExplainer only when needed."""

        if self.explainer is not None:
            return self.explainer

        if self.explanation_initialization_error is not None:
            raise RuntimeError(
                self.explanation_initialization_error
            )

        try:
            import shap

            self.explainer = shap.TreeExplainer(self.model)
            return self.explainer

        except Exception as exc:
            self.explanation_initialization_error = (
                f"Could not initialize SHAP TreeExplainer: {exc}"
            )
            raise RuntimeError(
                self.explanation_initialization_error
            ) from exc

    @staticmethod
    def _extract_positive_class_shap_values(shap_output, feature_count):
        """Normalize common SHAP output formats for binary classification."""

        if hasattr(shap_output, "values"):
            shap_output = shap_output.values

        if isinstance(shap_output, list):
            if len(shap_output) >= 2:
                shap_output = shap_output[1]
            elif len(shap_output) == 1:
                shap_output = shap_output[0]
            else:
                raise ValueError(
                    "SHAP returned an empty list."
                )

        values = np.asarray(shap_output)

        # Common shape: (rows, features, classes).
        if (
            values.ndim == 3
            and values.shape[0] == 1
            and values.shape[1] == feature_count
        ):
            if values.shape[2] >= 2:
                values = values[0, :, 1]
            elif values.shape[2] == 1:
                values = values[0, :, 0]

        # Common shape: (rows, features).
        elif values.ndim == 2:
            if values.shape[0] != 1:
                raise ValueError(
                    "Expected SHAP values for one transaction."
                )
            values = values[0]

        elif values.ndim == 1:
            pass

        else:
            raise ValueError(
                f"Unsupported SHAP output shape: {values.shape}"
            )

        values = np.asarray(values, dtype=float).reshape(-1)

        if len(values) != feature_count:
            raise ValueError(
                "SHAP feature count mismatch: "
                f"received {len(values)}, expected {feature_count}."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "SHAP returned non-finite feature contributions."
            )

        return values

    def _calculate_explanations(self, processed):
        """Calculate feature contributions for the positive fraud class."""

        try:
            explainer = self._get_explainer()
            shap_output = explainer.shap_values(processed)

            contributions = self._extract_positive_class_shap_values(
                shap_output,
                len(self.feature_columns),
            )

            feature_attributions = []

            for index, feature in enumerate(self.feature_columns):
                contribution = float(contributions[index])

                feature_attributions.append(
                    {
                        "feature": str(feature),
                        "value": float(processed.iloc[0, index]),
                        "shap_value": contribution,
                        "contribution": contribution,
                        "direction": (
                            "increases model output"
                            if contribution > 0
                            else "decreases model output"
                            if contribution < 0
                            else "no contribution"
                        ),
                    }
                )

            positive = sorted(
                [
                    item for item in feature_attributions
                    if item["shap_value"] > 0
                ],
                key=lambda item: item["shap_value"],
                reverse=True,
            )[:TOP_EXPLANATION_FEATURES]

            negative = sorted(
                [
                    item for item in feature_attributions
                    if item["shap_value"] < 0
                ],
                key=lambda item: item["shap_value"],
            )[:TOP_EXPLANATION_FEATURES]

            return {
                "feature_attributions": feature_attributions,
                "top_positive_features": positive,
                "top_negative_features": negative,
                "explanation_status": "available",
                "explanation_method": "SHAP TreeExplainer",
                "explanation_scale": (
                    "Raw model output; contributions are not "
                    "fraud probability percentage points."
                ),
                "explanation_error": None,
            }

        except Exception as exc:
            # Prediction remains available even if SHAP is incompatible.
            return {
                "feature_attributions": [],
                "top_positive_features": [],
                "top_negative_features": [],
                "explanation_status": "unavailable",
                "explanation_method": "SHAP TreeExplainer",
                "explanation_scale": (
                    "Raw model output; contributions are not "
                    "fraud probability percentage points."
                ),
                "explanation_error": str(exc),
            }

    def assess_risk(self, transaction_features):
        """Return risk prediction and feature explanations for one transaction."""

        processed = self._preprocess(transaction_features)

        probability = float(
            self.model.predict_proba(processed)[0, 1]
        )

        prediction = int(
            probability >= self.threshold
        )

        result = {
            "fraud_probability": probability,
            "risk_score": probability * 100.0,
            "prediction": prediction,
            "classification": (
                "Fraud" if prediction == 1 else "Legitimate"
            ),
            "classification_threshold": self.threshold,
        }

        result.update(
            self._calculate_explanations(processed)
        )

        return result
