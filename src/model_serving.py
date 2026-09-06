import os
import json
import joblib
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from http.server import HTTPServer, BaseHTTPRequestHandler

from src.features import bitcoinheist_features
from src.risk_scoring import RiskAssessment

DEFAULT_EXPORT_DIR = "exported_models"


class ModelRegistry:
    """
    Registry & Production Loader for exported ChainGuard models across Small-to-Big tiers.
    """

    def __init__(self, export_dir: str = DEFAULT_EXPORT_DIR):
        self.export_dir = export_dir
        self.tabular_models = {}
        self.gnn_models = {}
        self.scaler = None
        self.feature_cols = None
        self.registry_metadata = {}

    def load_registry(self):
        os.makedirs(self.export_dir, exist_ok=True)
        meta_path = os.path.join(self.export_dir, "model_registry.json")

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.registry_metadata = json.load(f)

        # Load scaler
        scaler_path = os.path.join(self.export_dir, "tabular_scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # Load tabular models
        for filename in os.listdir(self.export_dir):
            full_path = os.path.join(self.export_dir, filename)
            if filename.endswith(".joblib") and filename != "tabular_scaler.joblib":
                data = joblib.load(full_path)
                model_key = filename.replace(".joblib", "")
                self.tabular_models[model_key] = data.get("model")
                if not self.feature_cols and "feature_cols" in data:
                    self.feature_cols = data["feature_cols"]

            elif filename.endswith(".ptc") or filename == "elliptic_graphsage.pt":
                try:
                    if filename.endswith(".ptc"):
                        gnn_model = torch.jit.load(full_path)
                    else:
                        gnn_model = torch.load(full_path)
                    model_key = filename.replace(".ptc", "").replace(".pt", "")
                    self.gnn_models[model_key] = gnn_model
                except Exception as e:
                    print(f"[ModelRegistry] Warning loading GNN {filename}: {e}")

        print(f"[ModelRegistry] Loaded {len(self.tabular_models)} tabular models, {len(self.gnn_models)} GNN models from {self.export_dir}")


class InferenceEngine:
    """
    High-throughput Production Inference Engine.
    Exposes low-latency predictions across small, medium, large, and deep models.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.registry.load_registry()
        self.risk_assessment = RiskAssessment()

    def predict_ransomware(
        self,
        address_data: Dict[str, Any],
        model_tier: str = "large"
    ) -> Dict[str, Any]:
        df = pd.DataFrame([address_data])
        df_feat = bitcoinheist_features(df)

        feat_cols = self.registry.feature_cols or [
            'length', 'weight', 'count', 'looped', 'neighbors', 'income',
            'income_per_neighbor', 'income_per_count', 'loop_ratio',
            'weight_per_count', 'length_per_count', 'log_income',
            'log_weight', 'log_count', 'log_neighbors'
        ]
        available = [c for c in feat_cols if c in df_feat.columns]
        X = df_feat[available].fillna(0).values

        # Select model key by tier
        target_key = "xgboost_large"
        for key in self.registry.tabular_models.keys():
            if model_tier.lower() in key:
                target_key = key
                break

        model = self.registry.tabular_models.get(target_key)
        if not model and self.registry.tabular_models:
            model = list(self.registry.tabular_models.values())[0]

        if model is None:
            # Fallback heuristic score
            income = float(address_data.get("income", 0.0))
            score = float(min(1.0, income / 1e8))
            return {
                "address": address_data.get("address", "unknown"),
                "model_used": "fallback_heuristic",
                "ransomware_risk_score": round(score, 4),
                "is_ransomware": int(score >= 0.5)
            }

        t0 = time.time()
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            risk_score = float(probs[0])
        else:
            preds = model.predict(X)
            risk_score = float(preds[0])

        latency_ms = round((time.time() - t0) * 1000.0, 3)

        return {
            "address": address_data.get("address", "unknown"),
            "model_used": target_key,
            "ransomware_risk_score": round(risk_score, 4),
            "is_ransomware": int(risk_score >= 0.5),
            "latency_ms": latency_ms
        }

    def predict_risk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        entity_id = payload.get("entity_id", "unknown_entity")
        ransomware_score = payload.get("ransomware_score")
        ransomware_family = payload.get("ransomware_family")
        gnn_score = payload.get("gnn_score")
        anomaly_score = payload.get("anomaly_score")
        peel_chain = payload.get("peel_chain_detected", False)
        fan_out = payload.get("fan_out_detected", False)

        return self.risk_assessment.evaluate_entity(
            entity_id=entity_id,
            ransomware_score=ransomware_score,
            ransomware_family=ransomware_family,
            gnn_score=gnn_score,
            anomaly_score=anomaly_score,
            peel_chain_detected=peel_chain,
            fan_out_detected=fan_out
        )


class ServingAPIHandler(BaseHTTPRequestHandler):
    """
    Lightweight REST API HTTP Handler for hosting exported ChainGuard models.
    """

    engine = None

    def _send_response_json(self, data: Dict[str, Any], status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def do_GET(self):
        if self.path in ["/health", "/"]:
            self._send_response_json({
                "status": "healthy",
                "service": "ChainGuard Model Serving REST API",
                "version": "1.0.0"
            })
        elif self.path == "/models":
            self._send_response_json({
                "tabular_models": list(ServingAPIHandler.engine.registry.tabular_models.keys()),
                "gnn_models": list(ServingAPIHandler.engine.registry.gnn_models.keys())
            })
        else:
            self._send_response_json({"error": "Endpoint not found"}, status=404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if self.path == "/predict/ransomware":
            res = ServingAPIHandler.engine.predict_ransomware(
                address_data=payload,
                model_tier=payload.get("model_tier", "large")
            )
            self._send_response_json(res)
        elif self.path == "/predict/risk":
            res = ServingAPIHandler.engine.predict_risk(payload)
            self._send_response_json(res)
        else:
            self._send_response_json({"error": "POST endpoint not found"}, status=404)


def run_serving_server(host: str = "0.0.0.0", port: int = 8080):
    ServingAPIHandler.engine = InferenceEngine()
    server_address = (host, port)
    httpd = HTTPServer(server_address, ServingAPIHandler)
    print(f"[ModelServing] ChainGuard REST API Server running at http://{host}:{port}/")
    return httpd
