results · PY
Copy

# export_results.py
# Run this ONCE before launching the dashboard:
#   py export_results.py
#
# Then launch the dashboard:
#   py -m streamlit run layer4_dashboard.py
 
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
 
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from community import best_partition
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layer1_ingestion import load_data, clean_data, engineer_features
from layer2_graph import stratified_sample, build_graph
 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
 
 
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
        degree_ratio = out_deg / in_deg if in_deg > 0 else 0.0
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
 
 
def run_graphsage(features_df, G, epochs=150):
    data, _, _ = build_pyg_data(G, features_df)
    n_clean = (features_df['is_fraud'] == 0).sum()
    n_fraud = (features_df['is_fraud'] == 1).sum()
    weight  = torch.tensor([1.0, n_clean / n_fraud], dtype=torch.float)
    model     = GraphSAGE(data.num_node_features, 64, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss(weight=weight)
    loss_history = []
    print(f"Training GraphSAGE for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        out        = model(data.x, data.edge_index)
        probs_all  = torch.exp(out)[:, 1].numpy()
        probs_test = probs_all[data.test_mask.numpy()]
        y_test     = data.y[data.test_mask].numpy()
    auc = roc_auc_score(y_test, probs_test)
    print(f"AUC: {auc:.4f}")
    features_df = features_df.copy()
    features_df['graphsage_probability'] = probs_all
    return features_df, loss_history, y_test, probs_test, auc
 
 
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
    raw = IsolationForest(contamination=0.2, random_state=42).fit_predict(X)
    features_df = features_df.copy()
    features_df['anomaly_signal'] = (raw == -1).astype(int)
    return features_df
 
 
def compute_ensemble(features_df, threshold=0.5):
    max_fr = features_df['community_fraud_rate'].max()
    features_df['community_signal'] = (
        features_df['community_fraud_rate'] / max_fr if max_fr > 0 else 0.0
    )
    features_df['ensemble_score'] = (
        0.60 * features_df['graphsage_probability'] +
        0.25 * features_df['community_signal'] +
        0.15 * features_df['anomaly_signal']
    )
    flagged      = features_df[features_df['ensemble_score'] > threshold].copy()
    confirmed    = flagged[flagged['is_fraud'] == 1]
    total_fraud  = int(features_df['is_fraud'].sum())
    precision    = len(confirmed) / len(flagged)  if len(flagged) > 0  else 0.0
    recall       = len(confirmed) / total_fraud   if total_fraud > 0   else 0.0
    return features_df.sort_values('ensemble_score', ascending=False), flagged, precision, recall
 
 
def build_pr_curve(results_df):
    total_fraud = int(results_df['is_fraud'].sum())
    curve = []
    for t in np.round(np.arange(0.20, 0.96, 0.02), 2):
        f = results_df[results_df['ensemble_score'] > t]
        c = f[f['is_fraud'] == 1]
        p = len(c) / len(f)       if len(f) > 0      else 0.0
        r = len(c) / total_fraud  if total_fraud > 0  else 0.0
        curve.append({'threshold': float(t), 'precision': float(p),
                      'recall': float(r), 'flagged': int(len(f))})
    return curve
 
 
def build_graph_export(G, results, flagged_set, top_n=80):
    top_nodes = list(results.head(top_n)['node'])
    keep = set(top_nodes)
    for node in top_nodes:
        if node in G:
            keep.update(list(G.predecessors(node))[:3])
            keep.update(list(G.successors(node))[:3])
    subG      = G.subgraph(keep).copy()
    score_map = dict(zip(results['node'], results['ensemble_score']))
    graph_nodes = [
        {'id': n, 'is_fraud': int(subG.nodes[n].get('is_fraud', 0)),
         'flagged': n in flagged_set, 'score': float(score_map.get(n, 0.0))}
        for n in subG.nodes()
    ]
    graph_edges = [{'source': s, 'target': d} for s, d in subG.edges()]
    return graph_nodes, graph_edges
 
 
if __name__ == "__main__":
    print("=" * 55)
    print("  NetTrace — Building dashboard export")
    print("=" * 55)
 
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
 
    sample = stratified_sample(df, fraud_ratio=0.2, max_size=50000)
    G      = build_graph(sample)
 
    features_df = engineer_node_features(G, sample)
 
    features_df, community_scores = run_clustering(G, features_df)
    cfm = dict(zip(community_scores['community'], community_scores['fraud_rate']))
    features_df['community_fraud_rate'] = features_df['community'].map(cfm).fillna(0)
 
    features_df = run_isolation_forest(features_df)
 
    features_df, loss_history, y_test, probs_test, auc = run_graphsage(
        features_df, G, epochs=150
    )
 
    THRESHOLD = 0.5
    results, flagged, precision, recall = compute_ensemble(features_df, THRESHOLD)
    flagged_set = set(flagged['node'])
 
    pr_curve = build_pr_curve(results)
    graph_nodes, graph_edges = build_graph_export(G, results, flagged_set)
 
    # Total money at risk from confirmed fraud senders
    amount_map = (
        sample.groupby('nameOrig')['amount'].sum()
        .reset_index().rename(columns={'nameOrig': 'node', 'amount': 'total_sent'})
    )
    flagged_amt = flagged.merge(amount_map, on='node', how='left')
    flagged_amt['total_sent'] = flagged_amt['total_sent'].fillna(0)
    total_at_risk = float(
        flagged_amt[flagged_amt['is_fraud'] == 1]['total_sent'].sum()
    )
 
    export = {
        'results':            results,
        'flagged':            flagged,
        'flagged_with_amounts': flagged_amt,
        'total_accounts':     int(G.number_of_nodes()),
        'total_transactions': int(G.number_of_edges()),
        'total_fraud_nodes':  int(results['is_fraud'].sum()),
        'total_at_risk':      total_at_risk,
        'precision':          float(precision),
        'recall':             float(recall),
        'auc':                float(auc),
        'threshold':          THRESHOLD,
        'weights':            {'graphsage': 0.60, 'community': 0.25, 'anomaly': 0.15},
        'pr_curve':           pr_curve,
        'loss_history':       loss_history,
        'y_test':             y_test,
        'probs_test':         probs_test,
        'community_scores':   community_scores.reset_index(drop=True),
        'graph_nodes':        graph_nodes,
        'graph_edges':        graph_edges,
    }
 
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(export, f)
 
    print(f"\n✓  Saved to: {out_path}")
    print(f"   Accounts analyzed:  {G.number_of_nodes():,}")
    print(f"   Flagged high-risk:  {len(flagged):,}")
    print(f"   Precision:          {precision:.1%}")
    print(f"   AUC:                {auc:.4f}")
    print(f"\nNow run:  py -m streamlit run layer4_dashboard.py")
