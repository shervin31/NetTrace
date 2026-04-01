# layer1_ingestion.py

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILEPATH = os.path.join(BASE_DIR, "data", "transactions.csv")


def load_data(filepath=FILEPATH):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} transactions")
    print(f"Columns: {list(df.columns)}")
    return df


def clean_data(df):
    cols = ['step', 'type', 'amount', 'nameOrig', 'nameDest',
            'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'isFraud']
    df = df[cols].copy()
    df = df[df['amount'] > 0]
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    print(f"Cleaned dataset:  {len(df):,} transactions remaining")
    print(f"Fraud cases:      {df['isFraud'].sum():,}")
    print(f"Clean cases:      {(df['isFraud'] == 0).sum():,}")
    return df


def simulate_stream(df, batch_size=500):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]


def engineer_features(df):
    df['balance_drop_ratio'] = df.apply(
        lambda r: r['amount'] / r['oldbalanceOrg']
        if r['oldbalanceOrg'] > 0 else 0, axis=1
    )
    df['dest_balance_increase'] = (
        df['newbalanceDest'] - df['oldbalanceDest']
    )
    df['amount_to_balance_ratio'] = df.apply(
        lambda r: r['amount'] / r['oldbalanceOrg']
        if r['oldbalanceOrg'] > 0 else 1.0, axis=1
    )
    df['dest_was_empty'] = (df['oldbalanceDest'] == 0).astype(int)

    print(f"\nFeatures engineered:")
    print(f"  balance_drop_ratio mean:        {df['balance_drop_ratio'].mean():.4f}")
    print(f"  dest_balance_increase mean:     {df['dest_balance_increase'].mean():.2f}")
    print(f"  amount_to_balance_ratio mean:   {df['amount_to_balance_ratio'].mean():.4f}")
    print(f"  dest_was_empty rate:            {df['dest_was_empty'].mean():.4f}")
    return df


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    print("\nSample transactions:")
    print(df.head(10))