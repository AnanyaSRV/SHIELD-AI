import pandas as pd
from dataset_paths import IEEE_TRANSACTION

print("IEEE transaction")
df = pd.read_csv(IEEE_TRANSACTION, nrows=1)
print(df.columns.tolist())

from dataset_paths import CREDIT_CARD

print("Credit card")
df = pd.read_csv(CREDIT_CARD, nrows=1)
print(df.columns.tolist())


from dataset_paths import PAYSIM
print("Paysim")
df = pd.read_csv(PAYSIM, nrows=1)
print(df.columns.tolist())