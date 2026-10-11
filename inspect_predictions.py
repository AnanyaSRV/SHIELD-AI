
import pandas as pd

files = [
    "fusion/lr_predictions.csv",
    "fusion/rf_predictions.csv",
    "fusion/xgb_predictions.csv",
]

dataframes = {}

for path in files:
    print("\n" + "=" * 55)
    print(path)

    df = pd.read_csv(path)
    dataframes[path] = df

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if "y_true" in df.columns:
        print("Missing labels:", df["y_true"].isna().sum())
        print("Label counts:")
        print(df["y_true"].value_counts(dropna=False).to_dict())
    else:
        print("ERROR: y_true column is missing")

    if "y_prob" in df.columns:
        print("Missing probabilities:", df["y_prob"].isna().sum())
        print("Probability range:",
              df["y_prob"].min(), "to", df["y_prob"].max())
    else:
        print("ERROR: y_prob column is missing")

print("\n" + "=" * 55)
print("PAIRWISE ALIGNMENT CHECK")

paths = list(dataframes)

for i in range(len(paths)):
    for j in range(i + 1, len(paths)):
        a = dataframes[paths[i]]
        b = dataframes[paths[j]]

        print(f"\n{paths[i]} vs {paths[j]}")
        print("Same row count:", len(a) == len(b))

        if "y_true" in a.columns and "y_true" in b.columns:
            if len(a) == len(b):
                print("Labels identical in row order:",
                      a["y_true"].reset_index(drop=True).equals(
                          b["y_true"].reset_index(drop=True)
                      ))
            else:
                print("Labels identical in row order: Not comparable")
