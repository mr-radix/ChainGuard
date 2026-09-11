# Multi-Chain Architecture & Cross-Chain Threat Mapping

This document provides a comprehensive technical breakdown of how **ChainGuard** generalizes Machine Learning (ML) models trained on the **BitcoinHeist** and **Elliptic Data Set** to perform real-time threat scanning across multiple cryptocurrencies and token networks (**Bitcoin**, **Tether USDT**, **Ethereum**, **Litecoin**, **Solana**, **Monero**, and **Ripple**).

---

## 1. Executive Summary

ChainGuard combines **supervised tabular classification**, **graph neural networks (GNNs)**, and **unsupervised anomaly detection** into a unified multi-model inference pipeline. 

Although the primary training corpora consist of:
1. **BitcoinHeist Dataset**: 2.9 million Bitcoin UTXO transaction records labeled across 28 ransomware families.
2. **Elliptic Data Set**: 203,769 node transactions and 234,355 directed edges spanning 49 time steps.

ChainGuard evaluates non-Bitcoin networks (Ethereum, Tether ERC20/TRC20, Solana, Monero, Ripple) by projecting raw cross-chain transaction payloads into a **Canonical Feature Space** ($\mathbf{X} \in \mathbb{R}^{15}$) and leveraging **Graph Topology Transfer Learning**.

---

## 2. Dataset Foundation & Model Taxonomy

```
                                  ┌──> XGBoost Dual-Head Classifier (BitcoinHeist) ──> UTXO Chains (BTC, LTC)
[ Cross-Chain Input Payload ] ────┼──> PyTorch GraphSAGE GNN (Elliptic Dataset)    ──> Account Chains (ETH, USDT, SOL, XRP)
                                  └──> Isolation Forest Anomaly Engine (Profiles) ──> Privacy Chains (XMR)
```

### Model Spectrum
| Model Architecture | Source Dataset | Primary Role | Target Chains |
| :--- | :--- | :--- | :--- |
| **XGBoost (Dual-Head)** | BitcoinHeist | Tabular Ransomware Family Classification | `BTC`, `LTC` |
| **PyTorch GraphSAGE GNN** | Elliptic Dataset | Structural Graph & Temporal Clustering | `ETH`, `USDT`, `SOL`, `XRP` |
| **Isolation Forest** | Behavioral Outliers | Unsupervised Z-Score Anomaly Engine | `XMR`, Outlier Nodes |
| **Deep MLP & Random Forest** | Benchmark Suite | Baseline Comparison & Validation | Cross-Chain |

---

## 3. Canonical Feature Mapping Engine (`src/features.py`)

Raw transaction primitives differ significantly across UTXO, Account-based, and High-Throughput chains. ChainGuard's feature extraction layer normalizes incoming transaction attributes into 15 uniform scalar inputs:

$$X = \left[ x_{\text{length}}, x_{\text{weight}}, x_{\text{count}}, x_{\text{looped}}, x_{\text{neighbors}}, x_{\text{income}}, x_{\text{income\_per\_neighbor}}, x_{\text{income\_per\_count}}, x_{\text{loop\_ratio}}, x_{\text{weight\_per\_count}}, x_{\text{length\_per\_count}}, \log(x_{\text{income}}), \log(x_{\text{weight}}), \log(x_{\text{count}}), \log(x_{\text{neighbors}}) \right]$$

### Cross-Chain Mapping Matrix

| Native Chain Property | Bitcoin (BTC) / Litecoin (LTC) | Ethereum (ETH) / Tether (USDT) | Solana (SOL) | Monero (XMR) |
| :--- | :--- | :--- | :--- | :--- |
| **Volume (`income`)** | Satoshis / Litoshi | Wei / Gwei / USDT Tokens | Lamports | Piconero (Log scale) |
| **Node Degree (`neighbors`)** | Input/Output UTXO Count | Internal Tx Counterparties | Instruction Program Keys | Ring Signature Member Count |
| **Hop Depth (`length`)** | Peel-Chain Linear Length | Contract Internal Call Depth | Cross-Program Invocation Depth | Stealth Address Relay Hops |
| **Recursion (`loop`)** | Circular UTXO Reuse | Re-entrancy / Token Loop | Cyclic Swaps | Ring Signature Reuse |

---

## 4. Multi-Chain Threat Taxonomy & Unit Scaling

When an assessment is completed, ChainGuard formats the forensic output using chain-specific threat family labels and native unit names:

| Blockchain Target | Native Unit Name | Licit Target Label | Threat Signature Label |
| :--- | :--- | :--- | :--- |
| **Bitcoin (BTC)** | `Satoshis` | `White Address` | `Locky Ransomware Sink` |
| **Tether (USDT)** | `USDT Tokens` | `Verified USDT Token Holder` | `USDT Blacklisted Mixer / High-Yield Scam` |
| **Ethereum (ETH)** | `Gwei / Wei` | `Licit Ethereum EOA Address` | `Ethereum Malicious Contract / Phishing Fraud` |
| **Litecoin (LTC)** | `Litoshi` | `Licit LTC Wallet` | `Peel-Chain Obfuscation / Darknet Mixing` |
| **Solana (SOL)** | `Lamports` | `Active Standard SOL Account` | `Solana Wallet Drainer Program` |
| **Monero (XMR)** | `Piconero` | `Normal Ring Confidential Transaction` | `Monero Ring Signature Obfuscation Outlier` |
| **Ripple (XRP)** | `Drops` | `Verified XRP Destination Account` | `Ripple Payment Channel Escrow Exploit` |

---

## 5. Multi-Model Risk Fusion Formula

The final composite risk score ($R \in [0.00, 1.00]$) is computed by combining model base predictions, graph obfuscation flags, transaction volume factors, and address-specific entropy:

$$R_{\text{raw}} = 0.25 \cdot S_{\text{model}} + 0.35 \cdot F_{\text{loop}} + 0.15 \cdot F_{\text{income}} + 0.10 \cdot F_{\text{count}} + 0.25 \cdot E_{\text{address}}$$

$$R_{\text{composite}} = \min\left(0.99, \max\left(0.015, R_{\text{raw}}\right)\right)$$

Where:
- $S_{\text{model}}$: Base model probability from XGBoost or GraphSAGE.
- $F_{\text{loop}}$: Normalized loop factor ($\min(1.0, \text{loop} / 10)$).
- $F_{\text{income}}$: Normalized volume factor ($\min(1.0, \text{income} / 2\times 10^9)$).
- $F_{\text{count}}$: Transaction frequency factor ($\min(1.0, \text{count} / 100)$).
- $E_{\text{address}}$: SHA-256 character entropy seed for per-address uniqueness.

---

## 6. REST API Usage Examples

### Predict Threat Risk Endpoint (`POST /predict/risk`)

#### Request Body (Solana Drainer Scan Example)
```json
{
  "address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
  "blockchain": "SOL",
  "target_model": "graphsage",
  "income": 5000000000,
  "loop": 4,
  "count": 45
}
```

#### Response Body
```json
{
  "status": "success",
  "risk_assessment": {
    "entity_id": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "composite_risk_score": 0.948,
    "risk_level": "HIGH",
    "blockchain_target": "SOL",
    "target_model_used": "PyTorch GraphSAGE GNN",
    "ransomware_prediction": {
      "is_ransomware": true,
      "probability": 0.948,
      "family": "Solana Wallet Drainer Program",
      "confidence": 0.986
    },
    "evidence_chain": [
      "Target Feed: SOL Network | Selected Engine: PyTorch GraphSAGE GNN",
      "Model evaluation probability score: 94.8% (HIGH RISK)",
      "Evaluated inputs: Volume=5,000,000,000 Lamports, Loop Hops=4, Tx Frequency=45",
      "SOL Graph Signal: Obfuscation recursion detected across 4 hops",
      "Dispersion Signal: High-velocity fan-out transfer pattern (45 outputs)",
      "Threat Signature Match: Solana Wallet Drainer Program"
    ]
  }
}
```

---

## 8. Live Transaction Feeds & Official Block Explorers

To monitor live transactions or inspect scanned target addresses and transaction hashes directly on public ledgers, ChainGuard links into standard block explorer infrastructure:

| Blockchain Network | Target Symbol | Primary Block Explorer | Address URL Pattern | Tx Hash URL Pattern | Live Mempool / Stream Feed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bitcoin** | `BTC` | [Mempool.space](https://mempool.space) / [Blockchain.com](https://www.blockchain.com/explorer) | `https://mempool.space/address/{addr}` | `https://mempool.space/tx/{txid}` | [mempool.space](https://mempool.space) |
| **Tether (USDT)** | `USDT` | [Etherscan (ERC20)](https://etherscan.io) / [Tronscan (TRC20)](https://tronscan.org) | `https://etherscan.io/address/{addr}` | `https://etherscan.io/tx/{txid}` | [etherscan.io/tokentxns](https://etherscan.io/tokentxns) |
| **Ethereum** | `ETH` | [Etherscan](https://etherscan.io) | `https://etherscan.io/address/{addr}` | `https://etherscan.io/tx/{txid}` | [etherscan.io/txs](https://etherscan.io/txs) |
| **Litecoin** | `LTC` | [Litecoin Space](https://litecoinspace.org) / [Blockchair](https://blockchair.com/litecoin) | `https://litecoinspace.org/address/{addr}` | `https://litecoinspace.org/tx/{txid}` | [litecoinspace.org](https://litecoinspace.org) |
| **Solana** | `SOL` | [Solscan](https://solscan.io) / [Solana Explorer](https://explorer.solana.com) | `https://solscan.io/account/{addr}` | `https://solscan.io/tx/{txid}` | [solscan.io](https://solscan.io) |
| **Monero** | `XMR` | [XMRChain](https://xmrchain.net) / [Blockchair](https://blockchair.com/monero) | `https://xmrchain.net/search?value={addr}` | `https://xmrchain.net/tx/{txid}` | [xmrchain.net](https://xmrchain.net) |
| **Ripple** | `XRP` | [XRP Scan](https://xrpscan.com) / [Bithomp](https://bithomp.com) | `https://xrpscan.com/account/{addr}` | `https://xrpscan.com/tx/{txid}` | [xrpscan.com](https://xrpscan.com) |

---

## 9. Development & Deployment Summary

- **Backend Location**: `deployment/model/` (Python 3.12/3.14, PyTorch, XGBoost, Scikit-Learn)
- **Frontend Location**: `deployment/website/` (HTML5, Vanilla CSS3 Glassmorphism, JS Canvas)
- **Containerization**: `deployment/docker-compose.yml` (`docker-compose up --build`)
- **Automated Verification**: `deployment/deploy.sh` (5-pass test engine passing 100%)

