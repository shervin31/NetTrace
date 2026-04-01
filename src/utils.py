# utils.py

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from community import best_partition
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import pandas as pd


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def engineer_node_features(G, df):
    account_stats = {}
    for _, row in df.iterrows():
        for acct in [row['nameOrig'], row['nameDest']]:
            if acct not in account_stats:
                account_stats[acct] = {
                    'balance_drops': [], 'amount_to_balance': [],
                    'dest_empty': [], 'dest_increases': []
                }
        s = row['nameOrig']
        account_stats[s]['balance_drops'].append(row['balance_drop_ratio'])
        account_stats[s]['amount_to_balance'].append(row['amount_to_balance_ratio'])
        account_stats[s]['dest_empty'].append(row['dest_was_empty'])
        account_stats[s]['dest_increases'].append(row['dest_balance_increase'])

    rows = []
    seen = set()
    for node in G.nodes():
        if node in seen:
            continue
        seen.add(node)
        in_deg   = G.in_degree(node)
        out_deg  = G.out_degree(node)
        is_fraud = G.nodes[node].get('is_fraud', 0)
        stats    = account_stats.get(node)
        degree_ratio = out_deg / in_deg if in_deg > 0 else float(out_deg)
        if stats and stats['balance_drops']:
            avg_bd  = np.mean(stats['balance_drops'])
            avg_ab  = np.mean(stats['amount_to_balance'])
            de_rate = np.mean(stats['dest_empty'])
            avg_di  = np.mean(stats['dest_increases'])
        else:
            avg_bd = avg_ab = de_rate = avg_di = 0.0
        rows.append({
            'node': node, 'in_degree': in_deg, 'out_degree': out_deg,
            'degree_ratio': round(degree_ratio, 4),
            'avg_balance_drop': round(avg_bd, 4),
            'avg_amount_to_balance': round(avg_ab, 4),
            'dest_empty_rate': round(de_rate, 4),
            'avg_dest_increase': round(avg_di, 4),
            'is_fraud': is_fraud
        })
    return pd.DataFrame(rows)


def build_pyg_data(G, features_df):
    feature_cols = [
        'in_degree', 'out_degree', 'degree_ratio',
        'avg_balance_drop', 'avg_amount_to_balance',
        'dest_empty_rate', 'avg_dest_increase'
    ]
    nodes = list(features_df['node'])
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[feature_cols].values)
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(features_df['is_fraud'].values, dtype=torch.long)
    edge_list = [
        [node_to_idx[s], node_to_idx[d]]
        for s, d in G.edges()
        if s in node_to_idx and d in node_to_idx
    ]
    edge_index = (
        torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        if edge_list else torch.zeros((2, 0), dtype=torch.long)
    )
    n = len(nodes)
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=42,
        stratify=features_df['is_fraud'].values
    )
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx]   = True
    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask), node_to_idx, scaler


def run_clustering(G, features_df):
    partition = best_partition(G.to_undirected(), randomize=False)
    features_df = features_df.copy()
    features_df['community'] = features_df['node'].map(partition)
    cs = (features_df.groupby('community')['is_fraud']
          .agg(['mean', 'count']).reset_index())
    cs.columns = ['community', 'fraud_rate', 'size']
    cs = cs[cs['size'] >= 5].sort_values('fraud_rate', ascending=False)
    return features_df, cs


def run_isolation_forest(features_df):
    feature_cols = [
        'in_degree', 'out_degree', 'degree_ratio',
        'avg_balance_drop', 'avg_amount_to_balance',
        'dest_empty_rate', 'avg_dest_increase'
    ]
    X = StandardScaler().fit_transform(features_df[feature_cols])
    raw = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
    features_df = features_df.copy()
    features_df['anomaly_signal'] = (raw == -1).astype(int)
    return features_df
