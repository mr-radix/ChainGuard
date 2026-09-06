# ChainGuard: Master Technical Specification & 10-Part Knowledge Base

---

## 📌 Table of Contents

- [Part 1: Executive Overview & System Architecture](#part-1-executive-overview--system-architecture)
- [Part 2: The Cryptocurrency Threat Landscape & Domain Challenges](#part-2-the-cryptocurrency-threat-landscape--domain-challenges)
- [Part 3: Multi-Chain Ingestion & Data Normalization Engine](#part-3-multi-chain-ingestion--data-normalization-engine)
- [Part 4: Feature Engineering & Behavioral Address Profiling](#part-4-feature-engineering--behavioral-address-profiling)
- [Part 5: Dual-Head Tabular Ransomware Classifier & Model Spectrum](#part-5-dual-head-tabular-ransomware-classifier--model-spectrum)
- [Part 6: Temporal Graph Neural Network for Illicit Node Detection](#part-6-temporal-graph-neural-network-for-illicit-node-detection)
- [Part 7: Graph Intelligence & Fund-Flow Traversal Engines](#part-7-graph-intelligence--fund-flow-traversal-engines)
- [Part 8: Unsupervised Anomaly Intelligence & Behavioral Drift](#part-8-unsupervised-anomaly-intelligence--behavioral-drift)
- [Part 9: Composite Risk Scoring & Evidence Fusion Engine](#part-9-composite-risk-scoring--evidence-fusion-engine)
- [Part 10: Production Serving & REST API Engine](#part-10-production-serving--rest-api-engine)

---

## Part 1: Executive Overview & System Architecture

### 1.1 What is ChainGuard?
**ChainGuard** is an enterprise-grade multi-chain cryptocurrency threat detection, fund-flow tracing, and forensic evidence fusion platform. It transforms raw, pseudonymous public blockchain transaction streams (**Bitcoin**, **Ethereum**, **Litecoin**) into actionable threat scores, topological graph visualizers, and court-ready forensic evidence trails.

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
             Investigator Dashboard (Streamlit) & CLI API
```

### 1.2 Core Operational Objectives
1. **Detect Known Threats**: Identify ransomware payout addresses and classify them into specific threat families (Locky, Cerber, CryptoLocker, etc.).
2. **Predict Illicit Graph Nodes**: Detect transaction-level illicit nodes on dynamic transaction graphs with zero temporal data leakage.
3. **Trace Fund Movements**: Perform multi-hop graph traversals and flag obfuscation patterns (peel chains, fan-out dispersion, fan-in consolidation).
4. **Uncover Zero-Day Anomalies**: Identify unlabelled behavioral anomalies using Isolation Forests and volume velocity spike tracking.
5. **Produce Explainable Reports**: Fuse all signals into explainable Risk Reports categorized into four distinct risk tiers without making unverified identity claims.

---

## Part 2: The Cryptocurrency Threat Landscape & Domain Challenges

### 2.1 The Pseudonymity Dilemma
Public blockchains record every transaction on a transparent, immutable distributed ledger. However, entities are identified solely by alphanumeric cryptographic public keys (e.g. `111K8kZAEnJg245r2cM6y9zgJGHZtJPy6`). Illicit actors exploit this pseudonymity using complex money-laundering patterns.

### 2.2 Critical Technical Challenges & ChainGuard Solutions

| Domain Challenge | Real-World Impact | ChainGuard Technical Solution |
|---|---|---|
| **Extreme Class Imbalance** | Illicit transactions constitute < 1.5% of total ledger volume. Standard ML yields high accuracy by ignoring criminal activity. | Cost-sensitive class reweighting (`scale_pos_weight = N_neg / N_pos`) in XGBoost and weighted BCE loss in PyTorch. |
| **Peel-Chain Laundering** | Criminals break funds into long chains of small transfers to pass under AML detection thresholds. | Heuristic graph pattern detector (`detect_peel_chain`) tracking linear change pass-throughs ($>65\%$) and peeled transfers ($\le 35\%$). |
| **Temporal Data Leakage in GNNs** | Standard random train/test splits allow graph models to aggregate features from future transactions. | Enforced chronological temporal splitting ($t \le 34$ train, $t \in [35, 39]$ val, $t \ge 40$ test). |
| **Heterogeneous Ledger Schemas** | Bitcoin uses UTXO inputs/outputs while Ethereum uses Account state balances. | Unified multi-chain normalization schema converting all streams into standard 7-column records. |

---

## Part 3: Multi-Chain Ingestion & Data Normalization Engine

**File**: `src/multichain_pipeline.py`

### 3.1 UTXO vs Account Model Mapping
ChainGuard ingests both UTXO-based chains (Bitcoin, Litecoin) and Account-based chains (Ethereum) into a single unified schema:

$$\text{Normalized Stream} = \{\text{chain}, \text{tx\_hash}, \text{from\_address}, \text{to\_address}, \text{amount}, \text{fee}, \text{timestamp}\}$$

```python
NORMALIZED_COLUMNS = ['chain', 'tx_hash', 'from_address', 'to_address', 'amount', 'fee', 'timestamp']
```

### 3.2 Ingestion & Normalization Functions
1. **`normalize_utxo_chain(raw_df, chain_name="BTC")`**:
   - Handles address-level inputs (e.g. BitcoinHeist data) by generating deterministic synthetic hashes: `tx_hash = btc_tx_0000100`.
   - Derives synthetic timestamps: $\text{timestamp} = (\text{year} - 2009) \times 31,536,000 + \text{day} \times 86,400$.
2. **`normalize_account_chain(raw_df, chain_name="ETH")`**:
   - Maps Ethereum account fields (`hash` $\rightarrow$ `tx_hash`, `from` $\rightarrow$ `from_address`, `to` $\rightarrow$ `to_address`, `value` $\rightarrow$ `amount`, `gas_price` $\rightarrow$ `fee`, `time` $\rightarrow$ `timestamp`).
3. **`merge_chains(chain_dfs)`**: Concatenates multi-chain normalized DataFrames and sorts chronologically.
4. **`cross_chain_entity_resolution(address_clusters, cross_chain_links)`**: Maps BTC, ETH, and LTC addresses to unified Entity IDs using shared exchange deposit addresses or bridge contract heuristics.

---

## Part 4: Feature Engineering & Behavioral Address Profiling

**File**: `src/features.py`

### 4.1 Feature Mathematical Formulations
Raw address metrics are transformed into behavioral ratio and log-scale features:

$$\text{income\_per\_neighbor} = \frac{\text{income}}{\text{neighbors} + 10^{-5}}$$

$$\text{income\_per\_count} = \frac{\text{income}}{\text{count} + 10^{-5}}$$

$$\text{loop\_ratio} = \frac{\text{looped}}{\text{count} + 10^{-5}}$$

$$\text{log\_income} = \log(1 + \max(0, \text{income}))$$

$$\text{temporal\_day} = (\text{year} - \text{min\_year}) \times 365 + \text{day}$$

### 4.2 Address Behavioral Profiler (`address_behavior_profile`)
Constructs comprehensive activity profiles from raw edge lists:
- **`fan_out_ratio`**: $\frac{\text{unique\_destinations}}{\text{sent\_count} + 10^{-5}}$
- **`fan_in_ratio`**: $\frac{\text{unique\_sources}}{\text{received\_count} + 10^{-5}}$
- **`active_duration_sec`**: $\text{max\_timestamp} - \text{min\_timestamp}$
- **`tx_velocity_per_hour`**: $\frac{\text{tx\_count}}{(\text{active\_duration\_sec} / 3600.0) + 1.0}$

---

## Part 5: Dual-Head Tabular Ransomware Classifier & Model Spectrum

**File**: `src/models/ransomware_model.py`

### 5.1 Dual-Head XGBoost Architecture
Designed specifically for the 2.9M address BitcoinHeist dataset:

```
                              Input Address Features (15 Cols)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          Head 1: Binary Ransomware Detector         Head 2: Family Triage Classifier
          (XGBoost, scale_pos_weight = N_neg/N_pos)  (XGBoost Multiclass, 11 Families)
                       │                                           │
                       ▼                                           ▼
             P(Is Ransomware) ∈ [0, 1]                    Predicted Family (Locky, Cerber...)
```

### 5.2 Small-to-Big Model Spectrum (`RansomwareModelSuite`)
Benchmarking suite across 5 model tiers:

| Tier | Model Architecture | Hyperparameters / Structure | Key Advantage |
|---|---|---|---|
| **Small** | `DecisionTreeClassifier` | Max depth: 5, Class weight: balanced | Low latency, fully interpretable rules |
| **Small** | `LogisticRegression` | Max iter: 500, Class weight: balanced | Fast baseline, micro-resource footprint |
| **Medium** | `RandomForestClassifier` | 100 estimators, Max depth: 10, Parallel `n_jobs=-1` | Robust ensemble, reduced variance |
| **Large** | `XGBClassifier` | 120 estimators, Max depth: 6, `scale_pos_weight` | Highest PR-AUC score on tabular data |
| **Deep** | `MLPClassifier` | 4 layers (128 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 1) | Deep feature representation learning |

---

## Part 6: Temporal Graph Neural Network for Illicit Node Detection

**File**: `src/models/elliptic_gnn.py`

### 6.1 PyTorch GraphSAGE Mean Aggregator (`GraphSAGELayer`)
GraphSAGE computes node embeddings by aggregating feature vectors from localized neighborhoods:

$$h_v^{(k)} = \text{ReLU} \left( W_{\text{self}} h_v^{(k-1)} + W_{\text{neigh}} \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)$$

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GraphSAGELayer, self).__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neigh_linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        out_self = self.self_linear(x)
        if edge_index.numel() == 0:
            return F.relu(out_self)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(N, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = torch.clamp(deg, min=1.0)
        neigh_x = x[src]
        aggregated = torch.zeros((N, x.size(1)), device=x.device, dtype=x.dtype)
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(neigh_x), neigh_x)
        aggregated = aggregated / deg.unsqueeze(1)
        return F.relu(out_self + self.neigh_linear(aggregated))
```

### 6.2 Temporal Train / Val / Test Split
To avoid data leakage across chronological time:
- **Train Set**: Timesteps $1 \le t \le 34$ ($159,481$ transactions)
- **Validation Set**: Timesteps $35 \le t \le 39$ ($23,712$ transactions)
- **Test Set**: Timesteps $40 \le t \le 49$ ($20,446$ transactions)

### 6.3 TorchScript Export (`export_torchscript`)
Compiles PyTorch GraphSAGE models into `.ptc` TorchScript modules using `torch.jit.trace` for high-throughput production deployment in C++ or Triton inference servers.

---

## Part 7: Graph Intelligence & Fund-Flow Traversal Engines

**File**: `src/graph_analysis.py`

### 7.1 UTXO Common-Input Co-signing Address Clustering (`UnionFind`)
Implements Disjoint Set Union (DSU) with path compression. Addresses co-signing inputs of a single UTXO transaction are proven to be controlled by the same wallet entity.

$$\text{find}(x): \text{parent}[x] = \text{find}(\text{parent}[x]) \quad \text{if } \text{parent}[x] \neq x$$

### 7.2 Multi-Hop BFS/DFS Traversal Engine (`trace_fund_flow`)
Executes breadth-first fund flow path tracing from a target seed address up to depth $N$. Tracks cumulative transfer amounts, timestamps, transaction hashes, and exact path sequences while preventing infinite cyclic loops.

### 7.3 Peel-Chain Pattern Detector (`detect_peel_chain`)
Identifies sequential transaction hops where:
1. Out-degree is $\ge 2$.
2. One destination receives $\ge 65\%$ of funds ($\text{main\_pass}$).
3. The secondary destination receives $\le 35\%$ of funds ($\text{peeled\_amount}$).

### 7.4 Fan-Out & Fan-In Pattern Detectors (`detect_fan_out_fan_in`)
- **Fan-Out**: Out-degree $\ge 4$ and $\text{out\_degree} > 2 \times \text{in\_degree}$ (Fund dispersion / Tumblers).
- **Fan-In**: In-degree $\ge 4$ and $\text{in\_degree} > 2 \times \text{out\_degree}$ (Fund consolidation / Exchanges).

---

## Part 8: Unsupervised Anomaly Intelligence & Behavioral Drift

**File**: `src/anomaly_detection.py`

### 8.1 Isolation Forest Anomaly Detector (`IsolationForestDetector`)
Fits an ensemble of random isolation trees on behavioral feature matrices (`tx_count`, `total_volume`, `net_flow`, `fan_out_ratio`, `fan_in_ratio`, `tx_velocity_per_hour`). Output decision scores are normalized into a $[0.0, 1.0]$ anomaly index.

### 8.2 Behavioral Drift & Velocity Surge Engine (`behavioral_change_score`)
Computes rolling z-score spikes in transaction amounts over a historical window $W$:

$$z = \frac{x_{\text{latest}} - \mu_{\text{past}}}{\sigma_{\text{past}} + 10^{-5}}$$

$$\text{Drift Score} = \frac{1}{1 + e^{-0.5 (z - 2.0)}}$$

---

## Part 9: Composite Risk Scoring & Evidence Fusion Engine

**File**: `src/risk_scoring.py`

### 9.1 Mathematical Risk Score Formulation
Fuses predictions from all model heads and graph pattern engines:

$$\text{Composite Risk Score} = w_r \cdot S_{\text{ransomware}} + w_g \cdot S_{\text{gnn}} + w_a \cdot S_{\text{anomaly}} + w_p \cdot S_{\text{pattern}}$$

Default weight configuration: $w_r = 0.35, w_g = 0.35, w_a = 0.15, w_p = 0.15$.

### 9.2 Risk Tier Classification Matrix

| Risk Tier | Score Range | Action / Description |
|---|---|---|
| **HIGH** | $0.75 - 1.00$ | Critical Threat: Flag for immediate asset freeze / FIU escalation. |
| **MEDIUM** | $0.50 - 0.74$ | Elevated Threat: Require Enhanced Due Diligence (EDD). |
| **LOW** | $0.25 - 0.49$ | Minor Anomaly: Monitor for additional activity hops. |
| **MINIMAL** | $0.00 - 0.24$ | Baseline: Standard legitimate wallet behavior. |

### 9.3 Automated Forensic Evidence Trail Generation
Outputs clear, human-readable evidence strings without making unverified identity claims:
- `"[Ransomware Model] High behavioral match to known ransomware payout patterns (Locky) (confidence: 94.2%)"`
- `"[GraphSAGE GNN] Transaction structural topology flagged as illicit flow (score: 88.5%)"`
- `"[Graph Engine] Active peel-chain obfuscation pattern detected (sequential peeling hops)"`

---

## Part 10: Production Serving & REST API Engine

**Files**: `src/model_serving.py`, `serve_models.py`

### 10.1 Production Serving Architecture & REST API
Exposes lightweight REST endpoints via Python's built-in `HTTPServer`:

```
POST /predict/ransomware ──► InferenceEngine ──► XGBoost / Joblib ──► JSON Response
POST /predict/risk       ──► RiskAssessment  ──► Fusion Report   ──► JSON Response
GET  /models             ──► ModelRegistry   ──► Loaded Artifacts──► JSON Response
GET  /health             ──► Healthcheck     ──► Status 200      ──► JSON Response
```

### 10.2 Inference Engine & Model Registry Workflows
- **`ModelRegistry`**: Loads joblib tabular models, scaler metadata, and TorchScript GNN modules from `exported_models/`.
- **`InferenceEngine`**: High-throughput production inference handler supporting low-latency predictions across small, medium, large, and deep models.
- **`ServingAPIHandler`**: Base HTTP handler receiving JSON payloads and returning risk scores, latency metrics, and evidence trails.


---

## Technical Q&A Summary

### Q: Why not use simple heuristics without machine learning?
**A**: Heuristics (like peel chains) only detect known static patterns. Machine learning (XGBoost & GraphSAGE) detects complex non-linear feature combinations and novel topological structures that heuristics miss. Fusing both yields optimal recall and precision.
