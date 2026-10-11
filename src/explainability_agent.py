import pandas as pd
import lightgbm as lgb
import shap

from sklearn.model_selection import train_test_split


class ExplainabilityAgent:

    def __init__(self):

        print("Loading IEEE-UFFS...")

        df = pd.read_csv("data/ieee_uffs.csv")

        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42
        )

        # ==========================================
        # CLASS WEIGHT
        # ==========================================
        raw_weight = (
            (y_train == 0).sum()
            /
            (y_train == 1).sum()
        )

        scale_pos_weight = min(
            raw_weight,
            10
        )

        print("\nRaw Weight:",
              round(raw_weight, 2))

        print("Using Weight:",
              scale_pos_weight)

        # ==========================================
        # MODEL
        # ==========================================
        print("\nTraining explanation model...")

        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        self.model.fit(
            X_train,
            y_train
        )

        self.X_test = X_test.reset_index(
            drop=True
        )

        self.y_test = y_test.reset_index(
            drop=True
        )

        # ==========================================
        # SHAP EXPLAINER
        # ==========================================
        print("Creating SHAP explainer...")

        self.explainer = shap.TreeExplainer(
            self.model
        )

        print("Explainability Agent Ready.")

    def explain_transaction(
        self,
        index
    ):

        transaction = self.X_test.iloc[[index]]

        # ==========================================
        # SHAP VALUES
        # ==========================================
        shap_values = self.explainer.shap_values(
            transaction
        )

        prediction = (
            self.model.predict_proba(
                transaction
            )[0][1]
        )

        # ==========================================
        # SHAP COMPATIBILITY FIX
        # ==========================================
        if isinstance(shap_values, list):

            shap_scores = shap_values[1][0]

        else:

            shap_scores = shap_values[0]

        # ==========================================
        # FEATURE CONTRIBUTIONS
        # ==========================================
        contributions = pd.DataFrame({

            "feature":
                transaction.columns,

            "value":
                transaction.iloc[0].values,

            "shap":
                shap_scores

        })

        contributions["abs_shap"] = (
            contributions["shap"].abs()
        )

        contributions = contributions.sort_values(
            "abs_shap",
            ascending=False
        )

        # ==========================================
        # FEATURE EVIDENCE
        # ==========================================
        feature_evidence = []

        for _, row in contributions.head(3).iterrows():

            feature_evidence.append(

                f"{row['feature']} "
                f"(Value={row['value']:.4f}, "
                f"SHAP={row['shap']:.4f})"

            )

        # ==========================================
        # BEHAVIORAL EVIDENCE
        # ==========================================
        behavioral_evidence = []

        transaction_row = transaction.iloc[0]

        if (
            "is_high_amount" in transaction.columns
            and transaction_row["is_high_amount"] == 1
        ):
            behavioral_evidence.append(
                "High transaction amount"
            )

        if (
            "is_night_transaction" in transaction.columns
            and transaction_row["is_night_transaction"] == 1
        ):
            behavioral_evidence.append(
                "Night-time transaction"
            )

        if (
            "rapid_transaction" in transaction.columns
            and transaction_row["rapid_transaction"] == 1
        ):
            behavioral_evidence.append(
                "Rapid transaction behaviour"
            )

        # ==========================================
        # DECISION RECOMMENDATION
        # ==========================================
        if prediction >= 0.80:

            recommendation = "BLOCK"

        elif prediction >= 0.60:

            recommendation = "MONITOR"

        elif prediction >= 0.30:

            recommendation = "VERIFY"

        else:

            recommendation = "ALLOW"

        # ==========================================
        # FINAL REPORT
        # ==========================================
        return {

            "fraud_probability":
                round(prediction, 4),

            "feature_evidence":
                feature_evidence,

            "behavioral_evidence":
                behavioral_evidence,

            "recommended_action":
                recommendation

        }


if __name__ == "__main__":

    agent = ExplainabilityAgent()

    result = agent.explain_transaction(
        index=0
    )

    print("\n" + "=" * 60)
    print("SHIELD-AI EXPLANATION")
    print("=" * 60)

    print(
        "\nFraud Probability:",
        result["fraud_probability"]
    )

    print("\nFeature Evidence:")

    for evidence in result[
        "feature_evidence"
    ]:

        print("•", evidence)

    print("\nBehavioral Evidence:")

    if result[
        "behavioral_evidence"
    ]:

        for evidence in result[
            "behavioral_evidence"
        ]:

            print("•", evidence)

    else:

        print("• None detected")

    print(
        "\nRecommended Action:",
        result["recommended_action"]
    )