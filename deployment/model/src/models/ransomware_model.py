import os
import pickle
import time
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, auc
)
from src.features import bitcoinheist_features

DEFAULT_FEATURE_COLS = [
    'length', 'weight', 'count', 'looped', 'neighbors', 'income',
    'income_per_neighbor', 'income_per_count', 'loop_ratio',
    'weight_per_count', 'length_per_count', 'log_income',
    'log_weight', 'log_count', 'log_neighbors'
]


class RansomwareClassifier:
    """
    Dual-head XGBoost model for BitcoinHeist dataset:
    Head 1: Binary classification (Ransomware vs. White)
    Head 2: Multiclass family classification (Locky, Cerber, CryptoLocker, etc.)
    """

    def __init__(self, feature_cols: Optional[list] = None):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.binary_model = None
        self.family_model = None
        self.family_encoder = LabelEncoder()
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = bitcoinheist_features(df)
        available_cols = [c for c in self.feature_cols if c in df_feat.columns]
        return df_feat[available_cols].fillna(0)

    def fit(
        self,
        df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        random_state: int = 42
    ) -> Dict[str, Any]:
        df_feat = bitcoinheist_features(df)
        X = df_feat[self.feature_cols].fillna(0)
        y_binary = df_feat['is_ransomware'].values

        num_neg = (y_binary == 0).sum()
        num_pos = (y_binary == 1).sum()
        scale_pos_weight = (num_neg / max(num_pos, 1))

        print(f"[RansomwareClassifier] Training Binary Model (Neg: {num_neg:,}, Pos: {num_pos:,}, scale_pos_weight: {scale_pos_weight:.2f})...")
        self.binary_model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        self.binary_model.fit(X, y_binary)

        ransomware_mask = (y_binary == 1)
        if ransomware_mask.sum() > 0:
            X_family = X[ransomware_mask]
            y_family_str = df_feat.loc[ransomware_mask, 'label'].values
            y_family_enc = self.family_encoder.fit_transform(y_family_str)

            print(f"[RansomwareClassifier] Training Family Model on {len(X_family):,} ransomware samples ({len(self.family_encoder.classes_)} families)...")
            self.family_model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            self.family_model.fit(X_family, y_family_enc)

        self.is_fitted = True
        return self.evaluate(df if eval_df is None else eval_df)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        X = self._prepare_features(df)
        y_true = df['is_ransomware'].values

        t0 = time.time()
        y_probs = self.binary_model.predict_proba(X)[:, 1]
        latency_ms = ((time.time() - t0) / len(df)) * 1000.0 if len(df) > 0 else 0.0

        y_preds = (y_probs >= 0.5).astype(int)

        precision = precision_score(y_true, y_preds, zero_division=0)
        recall = recall_score(y_true, y_preds, zero_division=0)
        f1 = f1_score(y_true, y_preds, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5

        p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(r_curve, p_curve)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "latency_ms_per_sample": round(latency_ms, 4),
            "sample_count": len(df),
            "pos_count": int(y_true.sum())
        }

    def predict_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X = self._prepare_features(df)
        return self.binary_model.predict_proba(X)[:, 1]

    def predict_family(self, df: pd.DataFrame) -> Tuple[np.ndarray, list]:
        if not self.is_fitted or self.family_model is None:
            return np.array([]), []

        X = self._prepare_features(df)
        family_preds_enc = self.family_model.predict(X)
        family_probs = self.family_model.predict_proba(X)
        family_names = list(self.family_encoder.inverse_transform(family_preds_enc))
        return family_probs, family_names

    def save(self, checkpoint_dir: str):
        os.makedirs(checkpoint_dir, exist_ok=True)
        binary_path = os.path.join(checkpoint_dir, "ransomware_binary.pkl")
        family_path = os.path.join(checkpoint_dir, "ransomware_family.pkl")

        with open(binary_path, "wb") as f:
            pickle.dump({"model": self.binary_model, "feature_cols": self.feature_cols}, f)

        if self.family_model is not None:
            with open(family_path, "wb") as f:
                pickle.dump({"model": self.family_model, "encoder": self.family_encoder}, f)

    def load(self, checkpoint_dir: str):
        binary_path = os.path.join(checkpoint_dir, "ransomware_binary.pkl")
        family_path = os.path.join(checkpoint_dir, "ransomware_family.pkl")

        if os.path.exists(binary_path):
            with open(binary_path, "rb") as f:
                data = pickle.load(f)
                self.binary_model = data["model"]
                self.feature_cols = data["feature_cols"]

        if os.path.exists(family_path):
            with open(family_path, "rb") as f:
                data = pickle.load(f)
                self.family_model = data["model"]
                self.family_encoder = data["encoder"]

        self.is_fitted = self.binary_model is not None


class RansomwareModelSuite:
    """
    Small-to-Big Model Benchmark & Export Suite for Tabular Ransomware Classification:
    - Micro/Small: LogisticRegression, DecisionTree
    - Medium: RandomForestClassifier
    - Large: XGBoostClassifier
    - Deep Neural: Deep MLP (4-layer neural net)
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.models = {}
        self.scaler = StandardScaler()

    def _prepare(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df_feat = bitcoinheist_features(df)
        X = df_feat[self.feature_cols].fillna(0).values
        y = df_feat['is_ransomware'].values
        return X, y

    def train_and_benchmark(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        if test_df is None:
            test_df = train_df

        X_train, y_train = self._prepare(train_df)
        X_test, y_test = self._prepare(test_df)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight = num_neg / max(num_pos, 1)

        model_candidates = {
            "DecisionTree": (
                DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=random_state),
                False, "Small"
            ),
            "LogisticRegression": (
                LogisticRegression(class_weight='balanced', max_iter=500, random_state=random_state),
                True, "Small"
            ),
            "RandomForest": (
                RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=random_state, n_jobs=-1),
                False, "Medium"
            ),
            "XGBoost": (
                XGBClassifier(n_estimators=120, max_depth=6, scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=-1, eval_metric='logloss'),
                False, "Large"
            ),
            "Deep MLP NeuralNet": (
                MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=250, random_state=random_state),
                True, "Deep"
            )
        }

        results = []

        for name, (model, requires_scale, tier) in model_candidates.items():
            t0 = time.time()
            if requires_scale:
                model.fit(X_train_scaled, y_train)
                fit_time = time.time() - t0

                t_inf = time.time()
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
                inf_latency_ms = ((time.time() - t_inf) / len(X_test)) * 1000.0
            else:
                model.fit(X_train, y_train)
                fit_time = time.time() - t0

                t_inf = time.time()
                y_probs = model.predict_proba(X_test)[:, 1]
                inf_latency_ms = ((time.time() - t_inf) / len(X_test)) * 1000.0

            y_preds = (y_probs >= 0.5).astype(int)

            prec = precision_score(y_test, y_preds, zero_division=0)
            rec = recall_score(y_test, y_preds, zero_division=0)
            f1 = f1_score(y_test, y_preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else 0.5

            p_curve, r_curve, _ = precision_recall_curve(y_test, y_probs)
            pr_auc = auc(r_curve, p_curve)

            self.models[name] = model

            results.append({
                "Model": name,
                "Model Tier": tier,
                "Precision": float(prec),
                "Recall": float(rec),
                "F1 Score": float(f1),
                "PR-AUC": float(pr_auc),
                "ROC-AUC": float(roc_auc),
                "Latency (ms/sample)": round(inf_latency_ms, 4),
                "Train Time (s)": round(fit_time, 3)
            })

        return pd.DataFrame(results).sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    def export_models(self, export_dir: str) -> Dict[str, str]:
        """
        Exports trained models to Joblib / Pickle files and creates model_registry.json.
        """
        os.makedirs(export_dir, exist_ok=True)
        export_paths = {}

        for name, model in self.models.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            model_file = os.path.join(export_dir, f"{safe_name}.joblib")
            joblib.dump({"model": model, "feature_cols": self.feature_cols, "scaler": self.scaler}, model_file)
            file_size_kb = round(os.path.getsize(model_file) / 1024.0, 2)
            export_paths[name] = {
                "file_path": model_file,
                "file_size_kb": file_size_kb
            }

        scaler_file = os.path.join(export_dir, "tabular_scaler.joblib")
        joblib.dump(self.scaler, scaler_file)

        print(f"[RansomwareModelSuite] Exported {len(self.models)} models to {export_dir}")
        return export_paths
