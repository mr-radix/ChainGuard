import pytest
import os
import json
import urllib.request
import threading
import time
from serve_models import export_small_to_big_models
from src.model_serving import ModelRegistry, InferenceEngine, run_serving_server


def test_export_and_registry():
    if not os.path.exists("Dataset/Bitcoin Heist Ransomware Address/BitcoinHeistData.csv"):
        pytest.skip("Dataset file not available")

    # Run export
    export_small_to_big_models(sample_size=1000)

    assert os.path.exists("exported_models/model_registry.json")
    assert os.path.exists("exported_models/elliptic_graphsage.ptc")

    registry = ModelRegistry()
    registry.load_registry()

    assert len(registry.tabular_models) > 0
    assert len(registry.gnn_models) > 0


def test_inference_engine():
    engine = InferenceEngine()

    sample_address = {
        "address": "111K8kZAEnJg245r2cM6y9zgJGHZtJPy6",
        "income": 100050000.0,
        "count": 10,
        "neighbors": 5,
        "length": 18,
        "weight": 1.0,
        "looped": 2
    }

    res_small = engine.predict_ransomware(sample_address, model_tier="small")
    assert "ransomware_risk_score" in res_small
    assert "latency_ms" in res_small

    res_risk = engine.predict_risk({
        "entity_id": sample_address["address"],
        "ransomware_score": res_small["ransomware_risk_score"],
        "peel_chain_detected": True
    })
    assert "risk_tier" in res_risk
    assert "composite_risk_score" in res_risk


def test_serving_rest_api():
    port = 8899
    server = run_serving_server(port=port)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    time.sleep(0.5)

    try:
        # Health check
        req = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
        assert req.status == 200
        data = json.loads(req.read().decode("utf-8"))
        assert data["status"] == "healthy"

        # POST /predict/ransomware
        payload = json.dumps({
            "address": "111K8kZAEnJg245r2cM6y9zgJGHZtJPy6",
            "income": 500000.0,
            "count": 5,
            "model_tier": "small"
        }).encode("utf-8")

        post_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict/ransomware",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(post_req)
        assert resp.status == 200
        result = json.loads(resp.read().decode("utf-8"))
        assert "ransomware_risk_score" in result
    finally:
        server.shutdown()
