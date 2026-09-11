import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

DEFAULT_ANOMALY_COLS = [
    'tx_count', 'total_volume', 'net_flow', 'fan_out_ratio', 
    'fan_in_ratio', 'tx_velocity_per_hour', 'sent_amount_std'
]


class IsolationForestDetector:
    """
    Unsupervised Anomaly Detector using Isolation Forest on behavioral feature vectors.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, contamination: float = 0.05):
        self.feature_cols = feature_cols or DEFAULT_ANOMALY_COLS
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

    def _prepare_matrix(self, df: pd.DataFrame) -> np.ndarray:
        available = [c for c in self.feature_cols if c in df.columns]
        if not available:
            numeric_df = df.select_dtypes(include=[np.number])
            return numeric_df.fillna(0).values

        X = df[available].copy()
        for col in X.columns:
            if 'volume' in col or 'amount' in col or 'flow' in col:
                X[col] = np.log1p(np.maximum(0, X[col]))
        return X.fillna(0).values

    def fit(self, df: pd.DataFrame):
        X = self._prepare_matrix(df)
        if len(X) > 0:
            self.model.fit(X)
            self.is_fitted = True
            print(f"[IsolationForestDetector] Model fitted on {len(X):,} samples.")
        return self

    def predict_anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            if len(df) >= 10:
                self.fit(df)
            else:
                return np.zeros(len(df))

        X = self._prepare_matrix(df)
        raw_scores = self.model.decision_function(X)
        scores_normalized = 1.0 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-5
        )
        return np.clip(scores_normalized, 0.0, 1.0)


class LocalOutlierFactorDetector:
    """
    Unsupervised Local Outlier Factor (LOF) Detector for density-based anomaly scoring.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, contamination: float = 0.05):
        self.feature_cols = feature_cols or DEFAULT_ANOMALY_COLS
        self.contamination = contamination
        self.model = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination, novelty=True)
        self.is_fitted = False

    def _prepare_matrix(self, df: pd.DataFrame) -> np.ndarray:
        available = [c for c in self.feature_cols if c in df.columns]
        if not available:
            numeric_df = df.select_dtypes(include=[np.number])
            return numeric_df.fillna(0).values

        X = df[available].copy()
        for col in X.columns:
            if 'volume' in col or 'amount' in col or 'flow' in col:
                X[col] = np.log1p(np.maximum(0, X[col]))
        return X.fillna(0).values

    def fit(self, df: pd.DataFrame):
        X = self._prepare_matrix(df)
        if len(X) > 0:
            self.model.fit(X)
            self.is_fitted = True
        return self

    def predict_anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            if len(df) >= 10:
                self.fit(df)
            else:
                return np.zeros(len(df))

        X = self._prepare_matrix(df)
        raw_scores = self.model.decision_function(X)
        scores_normalized = 1.0 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-5
        )
        return np.clip(scores_normalized, 0.0, 1.0)


def behavioral_change_score(address_history: pd.DataFrame, window: int = 5) -> float:
    if address_history.empty or len(address_history) < 2:
        return 0.0

    df = address_history.sort_values('timestamp')
    amounts = df['amount'].values

    if len(amounts) < window:
        mean_past = np.mean(amounts[:-1])
        std_past = np.std(amounts[:-1]) + 1e-5
        latest = amounts[-1]
    else:
        past = amounts[-window:-1]
        mean_past = np.mean(past)
        std_past = np.std(past) + 1e-5
        latest = amounts[-1]

    z_score = (latest - mean_past) / std_past
    norm_score = 1.0 / (1.0 + np.exp(-0.5 * (z_score - 2.0)))
    return float(np.clip(norm_score, 0.0, 1.0))
