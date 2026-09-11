import pandas as pd
import numpy as np
from src.anomaly_detection import IsolationForestDetector, behavioral_change_score


def test_isolation_forest_detector():
    df = pd.DataFrame({
        'tx_count': [1, 2, 1, 3, 2, 500],
        'total_volume': [10.0, 15.0, 8.0, 20.0, 12.0, 1000000.0],
        'net_flow': [0.0, 1.0, -1.0, 2.0, 0.0, 500000.0],
        'fan_out_ratio': [0.5, 0.5, 0.5, 0.5, 0.5, 0.99],
        'fan_in_ratio': [0.5, 0.5, 0.5, 0.5, 0.5, 0.01],
        'tx_velocity_per_hour': [0.1, 0.1, 0.1, 0.1, 0.1, 100.0],
        'sent_amount_std': [1.0, 1.0, 1.0, 1.0, 1.0, 50.0]
    })

    detector = IsolationForestDetector(contamination=0.2)
    detector.fit(df)
    scores = detector.predict_anomaly_score(df)
    assert len(scores) == 6
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    # The outlier (index 5) should have higher anomaly score than normal points
    assert scores[5] > scores[0]


def test_behavioral_change_score():
    address_history = pd.DataFrame({
        'timestamp': [100, 200, 300, 400, 500],
        'amount': [10.0, 10.0, 11.0, 10.5, 500.0]
    })
    score = behavioral_change_score(address_history)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Significant spike at the end
