# layer2_graph.py

import pandas as pd
import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layer1_ingestion import load_data, clean_data


def stratified_sample(df, fraud_ratio=0.2, max_size=50000):
    """
    Balanced sample — inflates fraud to 20% so the model can learn.
    Takes all available fraud cases and samples clean transactions
    to match the ratio. Caps at 50,000 transactions total.
    """
    fraud = df[df['isFraud'] == 1]
    clean = df[df['isFraud'] == 0]

    n_fraud = len(fraud)
    n_clean = int(n_fraud * ((1 - fraud_ratio) / fraud_ratio))

    total = n_fraud + n_clean
    if total > max_size:
        scale = max_size / total
        n_fraud = int(n_fraud * scale)
        n_clean = int(n_clean * scale)

    fraud_sample = fraud.sample(n=n_fraud, random_state=42)
    clean_sample = clean.sample(n=n_clean, random_state=42)

    sample = pd.concat([fraud_sample, clean_sample]).reset_index(drop=True)

    print(f"Fraud transactions:  {n_fraud:,} ({round(n_fraud/len(sample)*100, 1)}%)")
    print(f"Clean transactions:  {n_clean:,} ({round(n_clean/len(sample)*100, 1)}%)")
    print(f"Total sample size:   {len(sample):,}")
    return sample


def build_graph(df):
    """
    Builds a directed graph from transactions. Each account is a node,
    each transaction is a directed edge. Critically — both sender AND
    receiver get tagged as fraud when a transaction is fraudulent.
    PaySim only labels the sender in isFraud, but the receiver is a
    mule account and should be labeled too. This fix is what allows
    GraphSAGE to learn from the full fraud network, not just half of it.
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        sender = row['nameOrig']
        receiver = row['nameDest']
        amount = row['amount']
        is_fraud = int(row['isFraud'])
        step = int(row['step'])

        if G.has_edge(sender, receiver):
            G[sender][receiver]['weight'] += amount
            G[sender][receiver]['count'] += 1
        else:
            G.add_edge(sender, receiver,
                       weight=amount, count=1, step=step)

        for acct, label in [(sender, is_fraud), (receiver, is_fraud)]:
            if acct not in G.nodes:
                G.add_node(acct, is_fraud=label)
            else:
                G.nodes[acct]['is_fraud'] = max(
                    G.nodes[acct].get('is_fraud', 0), label
                )

    return G


def get_graph_stats(G):
    fraud_nodes = sum(
        1 for n in G.nodes if G.nodes[n].get('is_fraud', 0) == 1)
    clean_nodes = G.number_of_nodes() - fraud_nodes
    print(f"\nGraph stats:")
    print(f"  Total nodes (accounts):     {G.number_of_nodes():,}")
    print(f"  Total edges (transactions): {G.number_of_edges():,}")
    print(f"  Fraud-labeled nodes:        {fraud_nodes:,}")
    print(f"  Clean-labeled nodes:        {clean_nodes:,}")
    print(f"  Is directed graph:          {nx.is_directed(G)}")


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    print("\nCreating stratified sample...")
    sample = stratified_sample(df, fraud_ratio=0.2, max_size=50000)
    print("\nBuilding graph...")
    G = build_graph(sample)
    get_graph_stats(G)