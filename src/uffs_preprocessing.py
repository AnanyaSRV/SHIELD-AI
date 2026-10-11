
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from dataset_paths import (
    CREDIT_CARD,
    PAYSIM,
    IEEE_TRANSACTION
)


def build_uffs(
    df,
    amount_col,
    time_col,
    target_col
):
    """
    Convert dataset into Unified Fraud Feature Space (UFFS v3 Stage 1 Round 1).
    """

    uffs = pd.DataFrame()

    # ==========================================================
    # ORIGINAL FEATURES
    # ==========================================================
    uffs["amount"] = df[amount_col].fillna(0)
    uffs["time_feature"] = df[time_col].fillna(0)

    # ==========================================================
    # STANDARDIZED FEATURES
    # ==========================================================
    amount_scaler = StandardScaler()

    uffs["normalized_amount"] = (
        amount_scaler.fit_transform(
            uffs[["amount"]]
        ).flatten()
    )

    time_scaler = StandardScaler()

    uffs["normalized_time"] = (
        time_scaler.fit_transform(
            uffs[["time_feature"]]
        ).flatten()
    )

    # ==========================================================
    # EXISTING BEHAVIORAL FEATURES
    # ==========================================================
    high_amount_threshold = (
        uffs["amount"].quantile(0.95)
    )

    uffs["is_high_amount"] = (
        uffs["amount"] >= high_amount_threshold
    ).astype(int)

    # Dataset-aware hour extraction
    if uffs["time_feature"].max() > 10000:
        hours = (
            (uffs["time_feature"] // 3600) % 24
        )
    else:
        hours = (
            uffs["time_feature"] % 24
        )

    uffs["is_night_transaction"] = (
        (hours <= 6) |
        (hours >= 22)
    ).astype(int)

    # ==========================================================
    # UFFS v2 FEATURES
    # ==========================================================

    # Log Amount
    uffs["log_amount"] = np.log1p(
        uffs["amount"]
    )

    # Amount Percentile
    uffs["amount_percentile"] = (
        uffs["amount"]
        .rank(pct=True)
    )

    # Amount Z-Score
    amount_std = uffs["amount"].std()

    if amount_std == 0:
        uffs["amount_zscore"] = 0
    else:
        uffs["amount_zscore"] = (
            (
                uffs["amount"]
                - uffs["amount"].mean()
            )
            / amount_std
        )

    # Amount-to-Mean Ratio
    mean_amount = (
        uffs["amount"].mean()
    )

    if mean_amount == 0:
        uffs["amount_to_mean_ratio"] = 0
    else:
        uffs["amount_to_mean_ratio"] = (
            uffs["amount"]
            / mean_amount
        )

    # Time Percentile
    uffs["time_percentile"] = (
        uffs["time_feature"]
        .rank(pct=True)
    )

    # Rapid Transaction Proxy
    median_time = (
        uffs["time_feature"].median()
    )

    uffs["rapid_transaction"] = (
        uffs["time_feature"]
        <= median_time
    ).astype(int)

    # Time Bucket
    uffs["time_bucket"] = pd.cut(
        hours,
        bins=[-1, 6, 12, 18, 24],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Extreme Amount
    extreme_threshold = (
        uffs["amount"].quantile(0.99)
    )

    uffs["is_extreme_amount"] = (
        uffs["amount"] >= extreme_threshold
    ).astype(int)

    # High Deviation Flag
    uffs["high_deviation"] = (
        abs(uffs["amount_zscore"]) >= 2
    ).astype(int)

    # ==========================================================
    # UFFS v3 ROUND 1 - TIME FEATURES
    # ==========================================================

    uffs["hour"] = hours

    uffs["is_weekend"] = (
        ((uffs["time_feature"] // 24) % 7 >= 5)
    ).astype(int)

    uffs["hour_sin"] = np.sin(
        2 * np.pi * hours / 24
    )

    uffs["hour_cos"] = np.cos(
        2 * np.pi * hours / 24
    )

    # ==========================================================
    # UFFS v3 ROUND 1 - AMOUNT FEATURES
    # ==========================================================

    uffs["amount_squared"] = (
        uffs["amount"] ** 2
    )

    uffs["amount_decile"] = pd.qcut(
        uffs["amount"],
        10,
        labels=False,
        duplicates="drop"
    )

    median_amount = (
        uffs["amount"].median()
    )

    if median_amount == 0:
        uffs["amount_to_median_ratio"] = 0
    else:
        uffs["amount_to_median_ratio"] = (
            uffs["amount"]
            / median_amount
        )

    Q1 = uffs["amount"].quantile(0.25)
    Q3 = uffs["amount"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    uffs["iqr_outlier"] = (
        (uffs["amount"] < lower) |
        (uffs["amount"] > upper)
    ).astype(int)

    # ==========================================================
    # UFFS v3 ROUND 1 - VELOCITY FEATURES
    # ==========================================================

    uffs["time_diff"] = (
        uffs["time_feature"]
        .diff()
        .fillna(0)
    )

    rapid_threshold = (
        uffs["time_diff"]
        .quantile(0.10)
    )

    uffs["very_rapid"] = (
        uffs["time_diff"]
        <= rapid_threshold
    ).astype(int)

    uffs["rolling_txn_count"] = (
        uffs["time_feature"]
        .rolling(
            window=10,
            min_periods=1
        )
        .count()
    )

    # ==========================================================
    # TARGET
    # ==========================================================

    uffs["target"] = (
        df[target_col]
        .reset_index(drop=True)
    )

    return uffs


# ==========================================================
# CREDIT CARD DATASET
# ==========================================================

print("=" * 60)
print("PROCESSING CREDIT CARD DATASET")
print("=" * 60)

credit_df = pd.read_csv(CREDIT_CARD)

credit_uffs = build_uffs(
    credit_df,
    amount_col="Amount",
    time_col="Time",
    target_col="Class"
)

print(credit_uffs.head())
print("\nShape:", credit_uffs.shape)

os.makedirs("data", exist_ok=True)

credit_uffs.to_csv(
    "data/credit_uffs.csv",
    index=False
)

print("\nSaved: data/credit_uffs.csv")


# ==========================================================
# PAYSIM DATASET
# ==========================================================

print("\n" + "=" * 60)
print("PROCESSING PAYSIM DATASET")
print("=" * 60)

paysim_df = pd.read_csv(PAYSIM)

paysim_uffs = build_uffs(
    paysim_df,
    amount_col="amount",
    time_col="step",
    target_col="isFraud"
)

print(paysim_uffs.head())
print("\nShape:", paysim_uffs.shape)

paysim_uffs.to_csv(
    "data/paysim_uffs.csv",
    index=False
)

print("\nSaved: data/paysim_uffs.csv")


# ==========================================================
# IEEE-CIS DATASET
# ==========================================================

print("\n" + "=" * 60)
print("PROCESSING IEEE-CIS DATASET")
print("=" * 60)

ieee_df = pd.read_csv(
    IEEE_TRANSACTION,
    usecols=[
        "TransactionAmt",
        "TransactionDT",
        "isFraud"
    ]
)

ieee_uffs = build_uffs(
    ieee_df,
    amount_col="TransactionAmt",
    time_col="TransactionDT",
    target_col="isFraud"
)

print(ieee_uffs.head())
print("\nShape:", ieee_uffs.shape)

ieee_uffs.to_csv(
    "data/ieee_uffs.csv",
    index=False
)

print("\nSaved: data/ieee_uffs.csv")

print("\n" + "=" * 60)
print("UFFS v3 STAGE 1 ROUND 1 COMPLETE")
print("=" * 60)

print("\nGenerated Datasets:")
print("data/credit_uffs.csv")
print("data/paysim_uffs.csv")
print("data/ieee_uffs.csv")

print("\nTotal Features:")
print(credit_uffs.shape[1] - 1, "fraud features")
print("+ target")
print("=", credit_uffs.shape[1], "columns total")