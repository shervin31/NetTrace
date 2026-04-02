# NetTrace 🔍  
A graph-based fraud detection system that identifies suspicious accounts by analyzing transaction networks instead of individual transactions.

---

## Project Overview
Fraud does not happen in isolation — it occurs through networks of accounts moving money in coordinated ways. Traditional systems analyze transactions one at a time, missing these patterns.

NetTrace models financial data as a graph:
- Accounts → nodes  
- Transactions → directed edges  

By analyzing both account behavior and network structure, the system detects fraudulent accounts that would be invisible to transaction-level models.

Dataset: PaySim (Kaggle)  
https://www.kaggle.com/datasets/mtalaltariq/paysim-data

---

## Features
🧠 Graph-based modeling  
- Builds a directed transaction graph using NetworkX  
- Captures relationships between accounts  

🤖 Machine learning (GraphSAGE)  
- Graph Neural Network that learns from both node features and neighbors  
- Classifies accounts based on network behavior  

📊 Behavioral feature engineering  
- Balance drain ratios  
- Transaction-to-balance relationships  
- Destination account patterns  

⚠️ Anomaly detection  
- Isolation Forest identifies statistically unusual accounts  

🔗 Community detection  
- Louvain clustering groups accounts into risk-based communities  

🎯 Ensemble scoring  
- Combines graph learning, anomaly detection, and community signals  
- Produces a final risk score per account  

📈 Interactive dashboard  
- Streamlit dashboard for exploring results and model performance :contentReference[oaicite:0]{index=0}  

---

## How It Works
1. **Data Processing**  
   Clean transaction data and engineer behavioral features  

2. **Graph Construction**  
   Convert transactions into a network of connected accounts  

3. **Detection System**  
   - GraphSAGE → learns network patterns  
   - Isolation Forest → detects anomalies  
   - Louvain → identifies risky clusters  

4. **Ensemble Scoring**  
   Combine all signals into a final fraud risk score  

5. **Visualization**  
   Export results and analyze them through an interactive dashboard :contentReference[oaicite:1]{index=1}  

---

## Installation
```bash
git clone https://github.com/shervin31/NetTrace.git
cd NetTrace
pip install -r requirements.txt
```

## Dataset Setup
Download the dataset from Kaggle and place it in:****
```bash
data/transactions.csv
```

## Usage
Run full pipeline
```bash
cd src
python layer3_detection.py
```

## Generate Dashboard Data
```bash
python export_results.py
```

## Launch Dashboard 
```bash
streamlit run layer4_dashboard.py
```

Technologies Used
Python, Pandas, NumPy, NetworkX, PyTorch Geometric, Scikit-learn, Streamlit, Plotly

## Authors
- [Shervin Zare](https://linkedin.com/in/shervin-zare)  
- [Aryan Kakkar](https://www.linkedin.com/in/aryan-kakkar-/)  
- [Aatmik Bhagat](https://www.linkedin.com/in/aatmik-bhagat/)  


