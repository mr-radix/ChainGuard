import numpy as np
from typing import Dict, List, Any, Optional

RISK_TIERS = {
    "MINIMAL": (0.0, 0.25),
    "LOW": (0.25, 0.50),
    "MEDIUM": (0.50, 0.75),
    "HIGH": (0.75, 1.0)
}


class RiskAssessment:
    """
    Composite Risk Scoring & Evidence Fusion Engine.
    Fuses outputs from ransomware ML, Elliptic GNN, Isolation Forest, and Graph Pattern Detectors.
    Produces an explainable evidence list alongside the composite score and risk tier.
    """

    def __init__(
        self,
        w_ransomware: float = 0.35,
        w_gnn: float = 0.35,
        w_anomaly: float = 0.15,
        w_pattern: float = 0.15
    ):
        self.w_ransomware = w_ransomware
        self.w_gnn = w_gnn
        self.w_anomaly = w_anomaly
        self.w_pattern = w_pattern

        # Normalize weights to sum to 1.0
        total_w = w_ransomware + w_gnn + w_anomaly + w_pattern
        self.w_ransomware /= total_w
        self.w_gnn /= total_w
        self.w_anomaly /= total_w
        self.w_pattern /= total_w

    def classify_tier(self, score: float) -> str:
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.50:
            return "MEDIUM"
        elif score >= 0.25:
            return "LOW"
        else:
            return "MINIMAL"

    def evaluate_entity(
        self,
        entity_id: str,
        ransomware_score: Optional[float] = None,
        ransomware_family: Optional[str] = None,
        gnn_score: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        peel_chain_detected: bool = False,
        fan_out_detected: bool = False,
        fan_in_detected: bool = False,
        behavioral_drift_score: Optional[float] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fuses all available signals into a single Risk Assessment report with explicit evidence.
        """
        evidence = []
        pattern_subscores = []

        # 1. Ransomware model signal
        r_score = ransomware_score if ransomware_score is not None else 0.0
        if ransomware_score is not None and ransomware_score > 0.3:
            family_str = f" ({ransomware_family})" if ransomware_family else ""
            evidence.append(
                f"[Ransomware Model] High behavioral match to known ransomware payout patterns{family_str} "
                f"(confidence: {ransomware_score * 100:.1f}%)"
            )

        # 2. Elliptic GNN signal
        g_score = gnn_score if gnn_score is not None else 0.0
        if gnn_score is not None and gnn_score > 0.3:
            evidence.append(
                f"[GraphSAGE GNN] Transaction structural topology flagged as illicit flow "
                f"(score: {gnn_score * 100:.1f}%)"
            )

        # 3. Anomaly detection signal
        a_score = anomaly_score if anomaly_score is not None else 0.0
        if anomaly_score is not None and anomaly_score > 0.5:
            evidence.append(
                f"[Isolation Forest] High behavioral outlier score relative to baseline transactions "
                f"(anomaly index: {anomaly_score:.2f})"
            )

        if behavioral_drift_score is not None and behavioral_drift_score > 0.6:
            evidence.append(
                f"[Behavioral Drift] Sudden activity or volume velocity spike detected "
                f"(z-score index: {behavioral_drift_score:.2f})"
            )
            a_score = max(a_score, behavioral_drift_score)

        # 4. Graph pattern signals
        if peel_chain_detected:
            pattern_subscores.append(0.8)
            evidence.append("[Graph Engine] Active peel-chain obfuscation pattern detected (sequential peeling hops)")

        if fan_out_detected:
            pattern_subscores.append(0.7)
            evidence.append("[Graph Engine] Rapid fan-out dispersion pattern detected (high out-degree burst)")

        if fan_in_detected:
            pattern_subscores.append(0.6)
            evidence.append("[Graph Engine] Fan-in consolidation pattern detected (multiple input sources to single sink)")

        p_score = float(np.mean(pattern_subscores)) if pattern_subscores else 0.0

        # Composite score computation
        composite_score = (
            self.w_ransomware * r_score +
            self.w_gnn * g_score +
            self.w_anomaly * a_score +
            self.w_pattern * p_score
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        risk_tier = self.classify_tier(composite_score)

        if not evidence:
            evidence.append("[Baseline] No anomalous behavioral or graph topology patterns detected")

        return {
            "entity_id": entity_id,
            "composite_risk_score": round(composite_score, 4),
            "risk_tier": risk_tier,
            "signals": {
                "ransomware_score": round(r_score, 4),
                "ransomware_family": ransomware_family,
                "gnn_score": round(g_score, 4),
                "anomaly_score": round(a_score, 4),
                "pattern_score": round(p_score, 4)
            },
            "flags": {
                "peel_chain": peel_chain_detected,
                "fan_out": fan_out_detected,
                "fan_in": fan_in_detected
            },
            "evidence_list": evidence,
            "metadata": additional_metadata or {}
        }
