from src.risk_scoring import RiskAssessment


def test_risk_assessment_high_risk():
    engine = RiskAssessment()
    report = engine.evaluate_entity(
        entity_id="111K8kZAEnJg245r2cM6y9zgJGHZtJPy6",
        ransomware_score=0.88,
        ransomware_family="Locky",
        gnn_score=0.92,
        anomaly_score=0.85,
        peel_chain_detected=True,
        fan_out_detected=True
    )

    assert report["risk_tier"] == "HIGH"
    assert report["composite_risk_score"] >= 0.75
    assert len(report["evidence_list"]) >= 3
    assert any("Locky" in ev for ev in report["evidence_list"])
    assert any("peel-chain" in ev for ev in report["evidence_list"])


def test_risk_assessment_minimal_risk():
    engine = RiskAssessment()
    report = engine.evaluate_entity(
        entity_id="normal_wallet_001",
        ransomware_score=0.02,
        gnn_score=0.01,
        anomaly_score=0.05,
        peel_chain_detected=False
    )

    assert report["risk_tier"] == "MINIMAL"
    assert report["composite_risk_score"] < 0.25
    assert "No anomalous behavioral or graph topology patterns detected" in report["evidence_list"][0]
