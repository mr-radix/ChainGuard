# 🚀 ChainGuard Deployment Guide: Render.com

This guide provides step-by-step instructions for deploying the **ChainGuard Multi-Chain Threat Detection & Forensic Tracing Platform** to **[Render](https://render.com/)**.

Official GitHub Repository: [`https://github.com/mr-radix/ChainGuard`](https://github.com/mr-radix/ChainGuard)

---

## 🏗️ Architecture Overview on Render

ChainGuard runs on Render using a high-availability, decoupled microservice architecture:

| Component | Render Service Type | Source Directory | Runtime / Tech | URL Pattern |
|---|---|---|---|---|
| **Model REST API Server** | **Web Service** | `deployment/model/` | Docker (`Python 3.12`, `PyTorch`, `Scikit-Learn`) | `https://chainguard-model-server.onrender.com` |
| **Web Dashboard UI** | **Static Site** | `deployment/website/src/` | HTML5, Vanilla CSS Glassmorphism, JS Canvas | `https://chainguard-web-dashboard.onrender.com` |

---

## ⚙️ Prerequisites

1. A **[Render Account](https://render.com/)** (Free tier supported).
2. A GitHub account with the repository pushed or cloned: [`https://github.com/mr-radix/ChainGuard`](https://github.com/mr-radix/ChainGuard).
3. Docker installed locally (if building custom container images prior to push).

---

## ⚡ Option A: 1-Click Automated Deployment (Render Blueprints)

ChainGuard includes a pre-configured Infrastructure-as-Code Blueprint specification ([`render.yaml`](../render.yaml)) at the repository root.

### Steps:
1. Log in to your **[Render Dashboard](https://dashboard.render.com/)**.
2. Click **New +** at the top right and select **Blueprint**.
3. Connect your GitHub account and select the repository: `mr-radix/ChainGuard`.
4. Render will automatically read `render.yaml` and provision both services:
   - `chainguard-model-server` (Docker Web Service on port 8080)
   - `chainguard-web-dashboard` (Static Web Site)
5. Click **Apply**. Render will automatically build, deploy, and issue free SSL/TLS certificates for both services.

---

## 🛠️ Option B: Manual Web Interface Setup

If you prefer configuring services manually via the Render UI:

### 1. Deploy the ML Model REST API (`chainguard-model-server`)

1. Go to **Render Dashboard** → **New +** → **Web Service**.
2. Connect repository `mr-radix/ChainGuard`.
3. Configure the service settings:
   - **Name**: `chainguard-model-server`
   - **Region**: Oregon (or nearest to your users)
   - **Branch**: `main`
   - **Runtime**: **Docker**
   - **Docker Command Context**: `./deployment/model`
   - **Dockerfile Path**: `./deployment/model/Dockerfile`
   - **Instance Type**: **Starter** (recommended for PyTorch memory allocation, 512MB–1GB RAM)
4. Set Environment Variables:
   - `PORT`: `8080`
   - `PYTHONUNBUFFERED`: `1`
5. Under **Advanced Settings**, configure Health Check:
   - **Health Check Path**: `/health`
6. Click **Create Web Service**.

### 2. Deploy the Web Dashboard UI (`chainguard-web-dashboard`)

1. Go to **Render Dashboard** → **New +** → **Static Site**.
2. Connect repository `mr-radix/ChainGuard`.
3. Configure settings:
   - **Name**: `chainguard-web-dashboard`
   - **Branch**: `main`
   - **Build Command**: `echo "Deploying ChainGuard UI"`
   - **Publish Directory**: `deployment/website/src`
4. Click **Create Static Site**.

---

## 🌐 Custom API Configuration (Production URL Binding)

The web dashboard (`deployment/website/src/app.js`) automatically detects whether it is running on `localhost` or on Render:

```javascript
const API_BASE_URL = window.CHAIN_GUARD_API_URL || 
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:8080' 
    : 'https://chainguard-model-server.onrender.com');
```

If your Model Server URL differs from `https://chainguard-model-server.onrender.com`, insert the following inline script block inside `deployment/website/src/index.html` within the `<head>` element:

```html
<script>
  window.CHAIN_GUARD_API_URL = "https://your-custom-model-server.onrender.com";
</script>
```

---

## 🧪 Verification & Health Checks

Once deployed, verify the endpoints:

### 1. Model Server Health Endpoint
```bash
curl -i https://chainguard-model-server.onrender.com/health
```
**Expected Response (`HTTP 200 OK`)**:
```json
{
  "status": "online",
  "models_loaded": {
    "bitcoin_heist_rf": true,
    "elliptic_gnn": true,
    "isolation_forest_anomaly": true
  },
  "supported_chains": ["BTC", "USDT", "ETH", "LTC", "SOL", "XMR", "XRP"]
}
```

### 2. Threat Scan Test Request
```bash
curl -X POST https://chainguard-model-server.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "chain": "ETH",
    "features": [0.0, 1500000000000000000.0, 1.0, 1.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }'
```

---

## 🔒 Security & CORS Best Practices

Render enables HTTPS/SSL by default. To restrict CORS on the model server to only accept requests from your static dashboard domain, update `serve_models.py` header configuration:

```python
self.send_header("Access-Control-Allow-Origin", "https://chainguard-web-dashboard.onrender.com")
```

---

## 📁 Repository Reference

- **GitHub Repository**: [`https://github.com/mr-radix/ChainGuard`](https://github.com/mr-radix/ChainGuard)
- **Render Blueprint Spec**: [`render.yaml`](../render.yaml)
- **Model Dockerfile**: [`deployment/model/Dockerfile`](../deployment/model/Dockerfile)
- **Website Dockerfile**: [`deployment/website/Dockerfile`](../deployment/website/Dockerfile)
