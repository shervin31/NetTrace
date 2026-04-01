# layer3_detection.py

import random
import pandas as pd
import numpy as np
import networkx as nx
import torch
from sklearn.metrics import classification_report, roc_auc_score
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layer1_ingestion import load_data, clean_data, engineer_features
from layer2_graph import stratified_sample, build_graph
from utils import (GraphSAGE, engineer_node_features, build_pyg_data,
                   run_clustering, run_isolation_forest)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def run_graphsage(features_df, G, epochs=150):
    """
    Trains 2-layer GraphSAGE and evaluates on held-out test set.
    """
    print("Building PyG data object...")
    data, node_to_idx, scaler = build_pyg_data(G, features_df)

    n_clean = (features_df['is_fraud'] == 0).sum()
    n_fraud = (features_df['is_fraud'] == 1).sum()
    weight = torch.tensor([1.0, n_clean / n_fraud], dtype=torch.float)

    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss(weight=weight)

    print(f"Training GraphSAGE for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs_all = torch.exp(out)[:, 1].numpy()
        probs_test = probs_all[data.test_mask.numpy()]
        y_test = data.y[data.test_mask].numpy()

    preds_test = (probs_test >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs_test)

    print("\nGraphSAGE TEST SET results (honest held-out):")
    print(classification_report(y_test, preds_test))
    print(f"GraphSAGE Test AUC: {auc:.4f}")

    features_df = features_df.copy()
    features_df['graphsage_probability'] = probs_all

    return features_df, model, data


def compute_ensemble_score(features_df):
    """
    Combines three signals into a final suspicion score.
    Uses a simple fixed ensemble setup.
    """
    max_fraud_rate = features_df['community_fraud_rate'].max()
    if max_fraud_rate > 0:
        features_df['community_signal'] = (
            features_df['community_fraud_rate'] / max_fraud_rate
        )
    else:
        features_df['community_signal'] = 0

    features_df['ensemble_score'] = (
        0.60 * features_df['graphsage_probability'] +
        0.25 * features_df['community_signal'] +
        0.15 * features_df['anomaly_signal']
    )

    threshold = 0.5
    flagged = features_df[features_df['ensemble_score'] > threshold]
    confirmed = flagged[flagged['is_fraud'] == 1]
    total_fraud = features_df['is_fraud'].sum()

    precision = len(confirmed) / len(flagged) if len(flagged) > 0 else 0
    recall = len(confirmed) / total_fraud if total_fraud > 0 else 0

    print(f"\nFINAL (threshold={threshold}):")
    print(f"Ensemble flagged {len(flagged):,} high-risk accounts")
    print(f"Of those, {len(confirmed):,} are confirmed fraud")
    print(f"Precision: {precision:.1%}    Recall: {recall:.1%}")

    return features_df.sort_values('ensemble_score', ascending=False)


if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    print("\nCreating stratified sample...")
    sample = stratified_sample(df, fraud_ratio=0.2, max_size=50000)

    print("\nBuilding graph...")
    G = build_graph(sample)

    print("\nEngineering node features...")
    features_df = engineer_node_features(G, sample)

    print("\nRunning Louvain clustering...")
    features_df, community_scores = run_clustering(G, features_df)

    community_fraud_map = dict(zip(
        community_scores['community'],
        community_scores['fraud_rate']
    ))
    features_df['community_fraud_rate'] = (
        features_df['community'].map(community_fraud_map).fillna(0)
    )

    print("\nRunning Isolation Forest...")
    features_df = run_isolation_forest(features_df)

    print("\nRunning GraphSAGE...")
    features_df, model, data = run_graphsage(features_df, G, epochs=150)

    print("\nComputing ensemble score...")
    results = compute_ensemble_score(features_df)

    print("\nTop flagged accounts:")
    print(results[[
        'node', 'ensemble_score', 'graphsage_probability',
        'avg_balance_drop', 'dest_empty_rate',
        'degree_ratio', 'is_fraud'
    ]].head(20))