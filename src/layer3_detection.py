# layer3_detection.py

import random
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from community import best_partition
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layer1_ingestion import load_data, clean_data, engineer_features
from layer2_graph import stratified_sample, build_graph

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class GraphSAGE(torch.nn.Module):
    """
    Two-layer GraphSAGE for node-level fraud classification.
    Each node sees its own features plus its 2-hop neighborhood.
    A mule account surrounded by other suspicious accounts gets
    pulled toward fraud even if its own features look borderline.
    Class-weighted loss handles the fraud/clean imbalance.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def engineer_node_features(G, df):
    """
    Computes 7 behavioral features per node.
    1. in_degree — accounts sending money in
    2. out_degree — accounts receiving money out
    3. degree_ratio — out/in ratio
    4. avg_balance_drop — proportion of balance sent per transaction
    5. avg_amount_to_balance — amount relative to opening balance
    6. dest_empty_rate — how often sends to empty accounts
    7. avg_dest_increase — average balance change at destination
    """
    account_stats = {}

    for _, row in df.iterrows():
        sender = row['nameOrig']
        receiver = row['nameDest']

        for acct in [sender, receiver]:
            if acct not in account_stats:
                account_stats[acct] = {
                    'balance_drops': [],
                    'amount_to_balance': [],
                    'dest_empty': [],
                    'dest_increases': []
                }

        account_stats[sender]['balance_drops'].append(row['balance_drop_ratio'])
        account_stats[sender]['amount_to_balance'].append(row['amount_to_balance_ratio'])
        account_stats[sender]['dest_empty'].append(row['dest_was_empty'])
        account_stats[sender]['dest_increases'].append(row['dest_balance_increase'])

    rows = []
    seen_nodes = set()

    for node in G.nodes():
        if node in seen_nodes:
            continue
        seen_nodes.add(node)

        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        is_fraud = G.nodes[node].get('is_fraud', 0)
        stats = account_stats.get(node, None)
        degree_ratio = out_deg / in_deg if in_deg > 0 else 0.0

        if stats and len(stats['balance_drops']) > 0:
            avg_balance_drop = np.mean(stats['balance_drops'])
            avg_amount_to_balance = np.mean(stats['amount_to_balance'])
            dest_empty_rate = np.mean(stats['dest_empty'])
            avg_dest_increase = np.mean(stats['dest_increases'])
        else:
            avg_balance_drop = 0.0
            avg_amount_to_balance = 0.0
            dest_empty_rate = 0.0
            avg_dest_increase = 0.0

        rows.append({
            'node': node,
            'in_degree': in_deg,
            'out_degree': out_deg,
            'degree_ratio': round(degree_ratio, 4),
            'avg_balance_drop': round(avg_balance_drop, 4),
            'avg_amount_to_balance': round(avg_amount_to_balance, 4),
            'dest_empty_rate': round(dest_empty_rate, 4),
            'avg_dest_increase': round(avg_dest_increase, 4),
            'is_fraud': is_fraud
        })

    return pd.DataFrame(rows)


def build_pyg_data(G, features_df):
    """
    Converts graph and features into a PyG Data object.
    Includes stratified 80/20 train/test split so AUC is
    evaluated on held-out nodes only — honest metrics.
    """
    feature_cols = [
        'in_degree', 'out_degree', 'degree_ratio',
        'avg_balance_drop', 'avg_amount_to_balance',
        'dest_empty_rate', 'avg_dest_increase'
    ]

    nodes = list(features_df['node'])
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[feature_cols].values)
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(features_df['is_fraud'].values, dtype=torch.long)

    edge_list = []
    for src, dst in G.edges():
        if src in node_to_idx and dst in node_to_idx:
            edge_list.append([node_to_idx[src], node_to_idx[dst]])

    if len(edge_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    n = len(nodes)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=features_df['is_fraud'].values
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    )

    print("PyG Data object:")
    print(f"  Nodes:        {data.num_nodes:,}")
    print(f"  Edges:        {data.num_edges:,}")
    print(f"  Features:     {data.num_node_features}")
    print(f"  Train nodes:  {train_mask.sum().item():,}")
    print(f"  Test nodes:   {test_mask.sum().item():,}")

    return data, node_to_idx, scaler


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


def run_clustering(G, features_df):
    """
    Louvain community detection. Finds tightly connected account
    clusters and scores each by fraud concentration.
    """
    G_undirected = G.to_undirected()
    partition = best_partition(G_undirected, randomize=False)

    features_df = features_df.copy()
    features_df['community'] = features_df['node'].map(partition)

    community_scores = (
        features_df.groupby('community')['is_fraud']
        .agg(['mean', 'count'])
        .reset_index()
    )
    community_scores.columns = ['community', 'fraud_rate', 'size']
    community_scores = community_scores[community_scores['size'] >= 5]
    community_scores = community_scores.sort_values('fraud_rate', ascending=False)

    high_risk = community_scores[community_scores['fraud_rate'] > 0]
    print(f"\nLouvain found {len(community_scores):,} communities, {len(high_risk):,} with fraud rate > 0")
    print(community_scores.head(10))

    return features_df, community_scores


def run_isolation_forest(features_df):
    """
    Isolation Forest — unsupervised anomaly detection.
    Finds statistically unusual accounts without any fraud labels.
    """
    feature_cols = [
        'in_degree', 'out_degree', 'degree_ratio',
        'avg_balance_drop', 'avg_amount_to_balance',
        'dest_empty_rate', 'avg_dest_increase'
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[feature_cols])

    iso = IsolationForest(contamination=0.2, random_state=42)
    raw = iso.fit_predict(X)

    features_df = features_df.copy()
    features_df['anomaly_signal'] = (raw == -1).astype(int)

    print(f"\nIsolation Forest flagged {features_df['anomaly_signal'].sum():,} anomalous accounts")
    return features_df


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