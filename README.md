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

## 👥 Author & Contributors

- **Lead Developer & Maintainer**: [@mr-radix](https://github.com/mr-radix)

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/mr-radix/ChainGuard/issues).

---

## 📜 License

This project is licensed under the **Non-Commercial Personal & Educational Use License**.
Permission is granted strictly for **personal, educational, testing, research, and demonstration purposes only**. Commercial use, commercial distribution, paid services, or integration into commercial products is strictly prohibited without prior written consent from the author.

See [LICENSE](LICENSE) for full details.

