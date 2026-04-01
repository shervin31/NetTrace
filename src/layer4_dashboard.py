import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

st.set_page_config(page_title="NetTrace Dashboard", layout="wide")

@st.cache_data
def load_data():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results.pkl')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)

D = load_data()

st.title("NetTrace")
st.caption("Fraud detection dashboard")

if D is None:
    st.error("results.pkl not found. Run: py export_results.py")
    st.stop()

required_keys = [
    'results', 'flagged', 'precision', 'recall', 'auc',
    'total_accounts', 'total_transactions', 'total_fraud_nodes',
    'loss_history', 'y_test', 'probs_test', 'pr_curve'
]

missing = [k for k in required_keys if k not in D]
if missing:
    st.error(f"results.pkl is missing keys: {missing}")
    st.stop()

results = D['results'].copy()
flagged = D['flagged'].copy()
precision = D['precision']
recall = D['recall']
auc = D['auc']
total_accounts = D['total_accounts']
total_transactions = D['total_transactions']
total_fraud_nodes = D['total_fraud_nodes']
threshold = D.get('threshold', 0.5)

page = st.sidebar.radio(
    "Page",
    ["Overview", "Flagged Accounts", "Model Performance", "Financial Impact", "How It Works"]
)

st.sidebar.write(f"Accounts: {total_accounts:,}")
st.sidebar.write(f"Transactions: {total_transactions:,}")
st.sidebar.write(f"AUC: {auc:.4f}")
st.sidebar.write(f"Precision: {precision:.1%}")
st.sidebar.write(f"Recall: {recall:.1%}")
st.sidebar.write(f"Threshold: {threshold:.2f}")

if page == "Overview":
    st.header("Overview")

    confirmed = len(flagged[flagged['is_fraud'] == 1])

    st.markdown(
        f"""
        <div style='font-size:22px; font-weight:600; margin-bottom:8px;'>
            NetTrace analyzed {total_accounts:,} accounts and surfaced {confirmed:,} confirmed fraud accounts.
        </div>
        <div style='font-size:18px; color:#b8c7d9; margin-bottom:28px;'>
            The system scans transaction behavior as a network, not just one transaction at a time,
            allowing it to detect coordinated fraud patterns more effectively.
        </div>
        """,
        unsafe_allow_html=True
    )
    false_pos = len(flagged[flagged['is_fraud'] == 0])
    missed = max(total_fraud_nodes - confirmed, 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts scanned", f"{total_accounts:,}")
    c2.metric("Fraud found", f"{confirmed:,}")
    c3.metric("Precision", f"{precision:.1%}")
    c4.metric("AUC", f"{auc:.4f}")

    st.markdown("### What NetTrace found")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        fig = go.Figure(data=[go.Pie(
            labels=["Confirmed fraud found", "False positives", "Missed fraud"],
            values=[confirmed, false_pos, missed],
            hole=0.0,   # full pie chart
            textinfo="label+percent",
            textposition="inside",
            marker=dict(colors=["#ef4444", "#f59e0b", "#334155"])
        )])

        fig.update_layout(
            title="Fraud Detection Outcome",
            height=520,
            showlegend=True,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=16)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.info(
            f"**Headline Result**\n\n"
            f"Out of **{total_accounts:,}** accounts analyzed, NetTrace identified "
            f"**{confirmed:,}** confirmed fraud accounts."
        )
        st.info(
            f"**Why it matters**\n\n"
            f"When the system raises an alert, it is correct about **{precision:.1%}** of the time. "
            f"That makes it useful for investigators who need higher-confidence leads."
        )
        st.info(
            f"**Model strength**\n\n"
            f"The model achieved an **AUC of {auc:.4f}**, meaning it is strong at separating "
            f"fraudulent accounts from clean accounts based on network behavior."
        )

    st.markdown("### Quick insights")

    i1, i2, i3 = st.columns(3)
    with i1:
        st.info(
            "**Network-based detection**\n\n"
            "NetTrace does not just score transactions individually. It looks at how accounts are connected, "
            "which helps reveal coordinated fraud."
        )
    with i2:
        st.info(
            "**High-confidence flags**\n\n"
            "The system is designed to produce useful alerts rather than flooding investigators "
            "with too many low-quality false alarms."
        )
    with i3:
        st.info(
            "**Communicable results**\n\n"
            "The output is meant to be explainable: how many accounts were scanned, how many fraud cases were found, "
            "and how strong the model performed overall."
        )

elif page == "Flagged Accounts":
    st.header("Flagged Accounts")

    min_score = st.slider("Minimum risk score", 0.30, 1.00, float(threshold), 0.01)
    fraud_only = st.checkbox("Show confirmed fraud only")

    view = results[results['ensemble_score'] >= min_score].copy()
    if fraud_only:
        view = view[view['is_fraud'] == 1]

    cols = [
        'node', 'ensemble_score', 'graphsage_probability',
        'avg_balance_drop', 'dest_empty_rate', 'degree_ratio', 'is_fraud'
    ]
    cols = [c for c in cols if c in view.columns]

    display = view[cols].head(200).copy()
    display['ensemble_score'] = display['ensemble_score'].round(3)
    display['is_fraud'] = display['is_fraud'].map({1: 'Yes', 0: 'No'})
    display = display.rename(columns={
        'node':                  'Account ID',
        'ensemble_score':        'Risk Score',
        'graphsage_probability': 'Network Risk (GNN)',
        'avg_balance_drop':      'Avg Balance Drained',
        'dest_empty_rate':       'Sends to Empty Accounts',
        'degree_ratio':          'Send/Receive Ratio',
        'is_fraud':              'Confirmed Fraud',
    })
    display = display.reindex(columns=['Account ID', 'Risk Score', 'Network Risk (GNN)', 'Avg Balance Drained', 'Sends to Empty Accounts', 'Send/Receive Ratio', 'Confirmed Fraud'])

    st.dataframe(display, use_container_width=True, height=600, hide_index=True)

elif page == "Model Performance":
    st.header("Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("AUC", f"{auc:.4f}")
    c2.metric("Precision", f"{precision:.1%}")
    c3.metric("Recall", f"{recall:.1%}")

    loss_history = D['loss_history']
    loss_df = pd.DataFrame({
        "Epoch": list(range(1, len(loss_history) + 1)),
        "Loss": loss_history
    })
    fig1 = px.line(loss_df, x="Epoch", y="Loss")
    fig1.update_layout(height=350)
    st.plotly_chart(fig1, use_container_width=True)

    fpr, tpr, _ = roc_curve(D['y_test'], D['probs_test'])
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {auc:.4f}"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random", line=dict(dash='dash')))
    roc_fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    pr_df = pd.DataFrame(D['pr_curve'])
    fig3 = px.line(pr_df, x="recall", y="precision", hover_data=["threshold", "flagged"])
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Financial Impact":
    st.header("Financial Impact")

    total_at_risk = D.get('total_at_risk', 0.0)
    flagged_amt = D.get('flagged_with_amounts', flagged).copy()

    fraud_accounts = flagged_amt[flagged_amt['is_fraud'] == 1]
    n_fraud = len(fraud_accounts)
    avg_per_account = total_at_risk / n_fraud if n_fraud > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total at risk", f"${total_at_risk:,.0f}")
    m2.metric("Confirmed fraud accounts", f"{n_fraud:,}")
    m3.metric("Avg amount per fraud account", f"${avg_per_account:,.0f}")

    st.markdown("### Top 20 fraud accounts by total sent")

    if 'total_sent' in flagged_amt.columns:
        top20 = (
            fraud_accounts[['node', 'total_sent']]
            .sort_values('total_sent', ascending=False)
            .head(20)
        )
        bar_fig = px.bar(
            top20, x='node', y='total_sent',
            labels={'node': 'Account', 'total_sent': 'Total Sent ($)'},
        )
        bar_fig.update_layout(
            height=450,
            xaxis_tickangle=-45,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.info("Amount data not available in results.pkl. Re-run export_results.py.")

elif page == "How It Works":
    st.markdown("<h1 style='font-size:36px'>How NetTrace Works</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; color:#666'>A graph-based system that detects fraud through network behavior, not just individual transactions.</p>", unsafe_allow_html=True)

    st.markdown("---")

    # Layer 1
    st.markdown("<h2 style='font-size:28px'>Layer 1 — Data & Features</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:18px'>
    We start with millions of financial transactions and focus on the types where fraud actually occurs.
    From each transaction, we extract behavioral signals like:
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style='font-size:18px'>
        <li>How much of an account’s balance is being drained</li>
        <li>Whether money is sent to empty accounts</li>
        <li>How often accounts send vs receive money</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:18px'>These features capture patterns that are common in fraudulent behavior.</p>", unsafe_allow_html=True)

    st.markdown("---")

    # Layer 2
    st.markdown("<h2 style='font-size:28px'>Layer 2 — Graph Construction</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:18px'>
    Instead of analyzing transactions one by one, we turn the system into a network:
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style='font-size:18px'>
        <li>Each account becomes a <b>node</b></li>
        <li>Each transaction becomes a <b>connection (edge)</b></li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:18px'>
    This allows us to detect coordinated behavior — for example, chains of accounts passing money quickly.
    Fraud often hides in these patterns, not in single transactions.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Layer 3
    st.markdown("<h2 style='font-size:28px'>Layer 3 — Detection Model</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:18px'>
    We use a Graph Neural Network called <b>GraphSAGE</b>.
    It learns from:
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style='font-size:18px'>
        <li>The account’s own behavior</li>
        <li>The behavior of nearby connected accounts</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:18px'>
    This means even if an account looks normal on its own, it can still be flagged if it is surrounded by suspicious activity.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:18px'>
    We also combine:
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style='font-size:18px'>
        <li><b>Anomaly detection</b> (finds unusual behavior)</li>
        <li><b>Community detection</b> (finds clusters of risky accounts)</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:18px'>
    These are combined into one final risk score per account.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Layer 4
    st.markdown("<h2 style='font-size:28px'>Layer 4 — Results</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:18px'>
    Accounts above a risk threshold are flagged as high-risk.
    The system prioritizes <b>precision</b>, meaning:
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style='font-size:18px'>
        <li>Fewer false alarms</li>
        <li>Higher confidence in each flagged account</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:18px'>
    This makes the output useful for investigators, who need high-quality leads instead of large volumes of noise.
    </p>
    """, unsafe_allow_html=True)