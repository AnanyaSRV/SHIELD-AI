import os
import pandas as pd

def load_training_data():
    """
    Finds and opens your training files, then merges them into one single table.
    """
    tx_path = os.path.join('data', 'train_transaction.csv')
    id_path = os.path.join('data', 'train_identity.csv')
    
    print("Reading train_transaction.csv... (This takes a few seconds because it's big)")
    transaction_df = pd.read_csv(tx_path)
    
    print("Reading train_identity.csv...")
    identity_df = pd.read_csv(id_path)
    
    print("Merging both files into one master training table...")
    merged_df = pd.merge(transaction_df, identity_df, on='TransactionID', how='left')
    
    print(f"📊 Done! Master table created with shape: {merged_df.shape}\n")
    return merged_df

def check_fraud_percentage(df):
    """
    Counts how many transactions are safe (0) vs fraudulent (1).
    """
    print("--- Checking Class Distribution ---")
    counts = df['isFraud'].value_counts()
    percentages = df['isFraud'].value_counts(normalize=True) * 100
    
    print(f"🔹 Legitimate Transactions (0): {counts[0]} rows ({percentages[0]:.2f}%)")
    print(f"🔹 Fraudulent Transactions (1): {counts[1]} rows ({percentages[1]:.2f}%)")

