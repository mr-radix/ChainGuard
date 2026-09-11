# ChainGuard Production Deployment Package

This directory contains the production-ready **ChainGuard** multi-chain threat detection and forensic tracing platform, split into modular `model/` (backend ML engine & REST API) and `website/` (frontend web dashboard) services with pinned dependency version specifications.

---

## 📂 Deployment Package Structure

```
deployment/
├── model/                               # Backend Model Server & ML Engine
│   ├── src/                             # Multi-chain GNN & Ransomware ML src
│   ├── tests/                           # Pytest automated test suite
│   ├── Dataset/                         # BitcoinHeist & Elliptic datasets
│   ├── checkpoints/                     # Model weight checkpoints
│   ├── exported_models/                 # Production compiled model artifacts
│   ├── train.py                         # Model training CLI
│   ├── serve_models.py                  # Production REST API server
│   ├── requirements.txt                 # PINNED Model dependency versions
│   └── Dockerfile                       # Model server container build file
├── website/                             # Frontend Web Dashboard Application
│   ├── src/                             # HTML, CSS, and JS dashboard source
│   │   ├── index.html                   # Dashboard HTML UI
│   │   ├── styles.css                   # Glassmorphic CSS design system
│   │   └── app.js                       # Interactive JS controller & graph canvas
│   ├── package.json                     # PINNED Node frontend dependency versions
│   ├── requirements.txt                 # PINNED Python website host versions
│   └── Dockerfile                       # Web dashboard container build file
├── docker-compose.yml                   # Unified multi-container orchestration
├── deploy.sh                            # Automated 5-pass verification script
└── README.md                            # Deployment documentation
```

---

## 📌 Pinned Dependency Specifications

### Model Backend (`model/requirements.txt`)
- `pandas==3.0.5`
- `numpy==2.5.2`
- `scikit-learn==1.9.0`
- `joblib==1.5.3`
- `xgboost==3.4.1`
- `torch==2.13.0`
- `networkx==3.6.1`
- `matplotlib==3.11.1`
- `seaborn==0.13.2`
- `pytest==9.1.1`
- `urllib3==2.7.0`
- `requests==2.34.2`

### Website Frontend (`website/package.json`)
- `http-server==14.1.1`
- `eslint==8.57.0`
- `Node.js >= 18.0.0`

---

## 🧪 Multi-Pass Verification & Testing

Before deploying, run the automated 5-pass verification script to test all backend models, REST API endpoints, and web dashboard assets:

```bash
bash deployment/deploy.sh
```

The script automatically executes:
1. **Pass 1**: Environment dependency check.
2. **Pass 2**: Full pytest test suite (`pytest model/tests/ -v`).
3. **Pass 3**: Model export & model registry compilation check.
4. **Pass 4**: Live test server launch on port 8089 & HTTP API smoke tests (`/health`, `/models`, `/predict/ransomware`, `/predict/risk`).
5. **Pass 5**: Website dashboard asset integrity validation.

---

## 🚀 Production Deployment Options

### Option A: Docker Compose (Recommended)

To build and launch both model server (port 8080) and web dashboard (port 3000) in isolated containers:

```bash
cd deployment
docker-compose up --build -d
```

Access the Web Dashboard at: `http://localhost:3000`  
Access the Model REST API at: `http://localhost:8080`

### Option B: Manual Bare-Metal Execution

1. **Launch Model REST API Server**:
   ```bash
   cd deployment/model
   pip install -r requirements.txt
   python3 serve_models.py --serve --port 8080
   ```

2. **Launch Website Dashboard**:
   ```bash
   cd deployment/website
   npm install
   npm start
   ```
