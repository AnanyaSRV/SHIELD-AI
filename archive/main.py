import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.dirname(__file__)
    )
)

from src.preprocessing import preprocess_data
from src.balancer import split_encode_and_oversample
from src.models import train_and_evaluate_models


def main():

    print("Starting Fraud Detection Pipeline")

    master_df = preprocess_data(
        "data/train_transaction.csv",
        "data/train_identity.csv"
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        encoder,
    ) = split_encode_and_oversample(
        master_df
    )

    train_and_evaluate_models(
        X_train,
        X_val,
        y_train,
        y_val,
    )


if __name__ == "__main__":
    main()