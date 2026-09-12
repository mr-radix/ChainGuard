# ChainGuard: Multi-Chain Cryptocurrency Threat Detection & Tracing Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-GNN-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tabular-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Non--Commercial-red.svg)](LICENSE)

**ChainGuard** is an enterprise-grade multi-chain cryptocurrency threat detection, fund-flow tracing, and forensic evidence fusion platform. It addresses the fundamental pseudonymity challenge of public blockchains (Bitcoin, Ethereum, Litecoin) by combining tabular machine learning, graph deep learning, unsupervised anomaly detection, and heuristic fund-flow pattern analysis.

---

## 🌟 Key Features

1. **Dual-Head Ransomware Classifier (BitcoinHeist):** Class-weighted XGBoost binary detector and 11-family multiclass triage (Locky, Cerber, CryptoLocker, etc.).
2. **Transaction-Level Illicit GNN (Elliptic):** PyTorch GraphSAGE node-level Graph Neural Network trained using strict temporal splitting (train <= step 34, val 35-39, test >= 40) to eliminate temporal data leakage.
3. **Graph Intelligence & Fund-Flow Tracing:**
   - UTXO Common-Input Co-signing Address Clustering (Union-Find).
   - Multi-hop BFS/DFS Fund Traversal engine.
   - Peel-Chain Obfuscation Pattern Detector (linear hops with incremental decay).
   - Fan-Out Dispersion & Fan-In Consolidation Pattern Detectors.
4. **Unsupervised Anomaly Intelligence:** Isolation Forest model on behavioral features + rolling z-score velocity spike analyzer.
5. **Composite Risk Scoring & Evidence Fusion:** Fuses all model and graph signals into an explainable Risk Report (MINIMAL, LOW, MEDIUM, HIGH) with supporting evidence lists without making unverified identity claims.
6. **Multi-Chain Normalization:** Ingests UTXO (BTC, LTC) and Account-based (ETH) transaction feeds into a unified schema.
7. **Small-to-Big Model Spectrum:** Benchmarking suite across 5 model tiers (DecisionTree, LogisticRegression, RandomForest, XGBoost, Deep MLP).
8. **CLI Trainer & REST Serving API:** Command-line model trainer (`train.py`) and model export & REST server (`serve_models.py`).

---

## 🏗️ System Architecture

```
                     Raw Multi-Chain Transaction Feeds
                         (BTC / ETH / LTC data)
                                   │
                                   ▼
                Chain-Agnostic Ingestion & Normalization
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     Address Clustering    Transaction Graph     Feature Engineering
    (UTXO Common-Input)    Construction (NX)     (Behavioral Ratios)
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │        Model Layer        │
                     │                           │
                     │ 1. Tabular XGBoost        │
                     │    (BitcoinHeist)         │
                     │                           │
                     │ 2. PyTorch GraphSAGE GNN  │
                     │    (Elliptic)             │
                     │                           │
                     │ 3. Isolation Forest       │
                     │    (Unsupervised Anomaly) │
                     └───────────────────────────┘
                                   │
                                   ▼
                  Graph-Based Fund-Flow Tracing Engine
           (BFS/DFS, Peel Chains, Fan-Out / Fan-In Detection)
                                   │
                                   ▼
                 Composite Risk Scoring & Evidence Fusion
                                   │
                                   ▼
                    CLI Trainer & REST Serving API
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/mr-radix/ChainGuard.git
cd ChainGuard
pip install -r requirements.txt
```

### 2. Automated Test Suite

Run unit tests across all project components:

```bash
python3 -m pytest tests/ -v
```

### 3. Model Training CLI

Train tabular ransomware and GNN models:

```bash
# Train Ransomware Classifier
python3 train.py --task ransomware --data-dir Dataset

# Train Elliptic GraphSAGE GNN
python3 train.py --task elliptic --data-dir Dataset --epochs 30

# Train All Models
python3 train.py --task all --data-dir Dataset
```

### 4. Model Export & REST Serving API

Export production models and launch the HTTP REST server:

```bash
# Export trained models to exported_models/
python3 serve_models.py --export-only --data-dir Dataset

# Export and launch HTTP REST Serving Server (Port 8080)
python3 serve_models.py --serve --port 8080
```

---

## 🔌 REST API Endpoints

When hosting via `python3 serve_models.py --serve --port 8080`, the platform exposes:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/models` | List loaded tabular and GNN models |
| `POST` | `/predict/ransomware` | Predict ransomware score & family for address payload |
| `POST` | `/predict/risk` | Evaluate multi-signal composite risk assessment report |

---

## 📊 Datasets Supported

| Dataset | Type | Scale | Target Task |
|---|---|---|---|
| **BitcoinHeist** | Tabular, Address-level | ~2.9M addresses | Ransomware detection & 11-family classification |
| **Elliptic** | Temporal Graph, Tx-level | 203K nodes, 234K edges | Transaction-level illicit node classification |

Dataset files are expected in:
- `Dataset/Bitcoin Heist Ransomware Address/BitcoinHeistData.csv`
- `Dataset/Elliptic/` (`elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_features.csv`)

---

## 📂 Project Structure

```
ChainGuard/
├── LICENSE                           # Official MIT Open Source License
├── README.md                         # Documentation & Quickstart (this document)
├── notes.md                          # 10-Part Master Technical Notes & Knowledge Base
├── requirements.txt                  # Python dependencies
├── train.py                          # CLI model training entrypoint
├── serve_models.py                  # CLI model export & HTTP REST API server
├── checkpoints/                      # Trained model checkpoints
│   ├── ransomware_binary.pkl
│   ├── ransomware_family.pkl
│   └── elliptic_gnn.pt
├── exported_models/                  # Production export directory
│   ├── model_registry.json           # Model registry metadata & spectrum benchmarks
│   ├── tabular_scaler.joblib         # StandardScaler artifact
│   ├── xgboost.joblib                # Exported XGBoost model
│   ├── elliptic_graphsage.ptc        # TorchScript compiled GNN module
│   └── ...
├── notebooks/
│   └── crypto_intel_pipeline.ipynb   # End-to-end Jupyter demonstration notebook
├── tests/                            # Pytest automated test suite
└── src/                              # Core Python package
    ├── data_loading.py               # Dataset loaders
    ├── features.py                   # Feature engineering & address profiles
    ├── graph_analysis.py             # Graph traversal & pattern detectors
    ├── anomaly_detection.py          # Isolation Forest & behavioral drift
    ├── risk_scoring.py               # Composite risk scoring & evidence fusion
    ├── multichain_pipeline.py        # UTXO & Account multi-chain normalization
    ├── model_serving.py              # Model Registry & REST API engine
    ├── phases/                       # Pipeline phase modules
    │   └── phase_03_eda.py           # Exploratory Data Analysis module
    └── models/                       # Machine Learning model definitions
        ├── ransomware_model.py       # XGBoost tabular model & spectrum suite
        └── elliptic_gnn.py           # PyTorch GraphSAGE GNN & TorchScript exporter
```

---

## 📚 References & Research Papers (Inspirations)

The architecture, machine learning models, graph algorithms, and forensic heuristics in **ChainGuard** were built upon and inspired by the following foundational research papers and open datasets:

1. **BitcoinHeist: Topological Feature Extraction Using Bitcoin Ransomware Data**
   - **Authors:** Cuneyt Gurcan Akcora, Yulia R. Gel, Murat Kantarcioglu
   - **Publication:** *IEEE Transactions on Knowledge and Data Engineering (TKDE)*, 2020
   - **Direct Links:** [arXiv:1906.07852](https://arxiv.org/abs/1906.07852) \| [IEEE Xplore (DOI: 10.1109/TKDE.2020.3012579)](https://doi.org/10.1109/TKDE.2020.3012579) \| [PDF](https://arxiv.org/pdf/1906.07852.pdf)
   - **ChainGuard Role:** Core foundation for address-level topological feature engineering and the dual-head XGBoost ransomware binary detector & 11-family triage classifier ([`src/models/ransomware_model.py`](src/models/ransomware_model.py)).

2. **Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics**
   - **Authors:** Mark Weber, Domenic Puzone, Bryan Bo Zhou, Peter Bell, Sergio A. Lopez-Rojas, et al. (IBM Research & MIT-IBM Watson AI Lab)
   - **Publication:** *KDD 2019 Workshop on Anomaly Detection in Finance*
   - **Direct Links:** [arXiv:1908.02591](https://arxiv.org/abs/1908.02591) \| [PDF Direct Download](https://arxiv.org/pdf/1908.02591.pdf)
   - **ChainGuard Role:** Direct inspiration for transaction-level illicit node prediction on dynamic graphs with strict chronological temporal splitting to eliminate data leakage ([`src/models/elliptic_gnn.py`](src/models/elliptic_gnn.py)).

3. **Inductive Representation Learning on Large Graphs (GraphSAGE)**
   - **Authors:** William L. Hamilton, Rex Ying, Jure Leskovec (Stanford University)
   - **Publication:** *Advances in Neural Information Processing Systems (NeurIPS)*, 2017
   - **Direct Links:** [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) \| [NeurIPS Proceedings](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs) \| [PDF](https://arxiv.org/pdf/1706.02216.pdf)
   - **ChainGuard Role:** Architecture powering the PyTorch inductive neighborhood mean-aggregator GNN layer (`GraphSAGELayer` in [`src/models/elliptic_gnn.py`](src/models/elliptic_gnn.py)).

4. **A Fistful of Bitcoins: Characterizing Payments Among Men with No Names**
   - **Authors:** Sarah Meiklejohn, Marjori Pomarole, Grant Jordan, Kirill Levchenko, Damon McCoy, Geoffrey M. Voelker, Stefan Savage
   - **Publication:** *ACM Internet Measurement Conference (IMC)*, 2013
   - **Direct Links:** [ACM Digital Library (DOI: 10.1145/2504730.2504747)](https://doi.org/10.1145/2504730.2504747) \| [PDF Paper Link](https://dreadref.github.io/papers/2013-meiklejohn-fistful.pdf)
   - **ChainGuard Role:** Foundation for UTXO Common-Input Co-signing Address Clustering (Disjoint Set Union) and heuristic fund-flow pattern detectors such as peel chains, fan-out dispersion, and fan-in consolidation ([`src/graph_analysis.py`](src/graph_analysis.py)).

5. **Isolation Forest**
   - **Authors:** Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou
   - **Publication:** *IEEE International Conference on Data Mining (ICDM)*, 2008
   - **Direct Links:** [IEEE Xplore (DOI: 10.1109/ICDM.2008.17)](https://doi.org/10.1109/ICDM.2008.17) \| [PDF Paper Link](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
   - **ChainGuard Role:** Underpins the unsupervised behavioral anomaly index engine for unlabelled zero-day threat discovery ([`src/anomaly_detection.py`](src/anomaly_detection.py)).

6. **XGBoost: A Scalable Tree Boosting System**
   - **Authors:** Tianqi Chen, Carlos Guestrin
   - **Publication:** *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 2016
   - **Direct Links:** [arXiv:1603.02754](https://arxiv.org/abs/1603.02754) \| [ACM Digital Library (DOI: 10.1145/2939672.2939785)](https://doi.org/10.1145/2939672.2939785) \| [PDF](https://arxiv.org/pdf/1603.02754.pdf)
   - **ChainGuard Role:** High-performance class-weighted gradient tree boosting engine used across tabular model spectrum benchmarks ([`src/models/ransomware_model.py`](src/models/ransomware_model.py)).

---

## 👥 Author & Contributors

- **Lead Developer & Maintainer**: [@mr-radix](https://github.com/mr-radix)

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/mr-radix/ChainGuard/issues).

---

## 📜 License

This project is licensed under the **Non-Commercial Personal & Educational Use License**.
Permission is granted strictly for **personal, educational, testing, research, and demonstration purposes only**. Commercial use, commercial distribution, paid services, or integration into commercial products is strictly prohibited without prior written consent from the author.

See [LICENSE](LICENSE) for full details.

