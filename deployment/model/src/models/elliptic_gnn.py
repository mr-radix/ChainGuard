import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Mean Aggregator Layer in PyTorch.
    h_v = ReLU(W_self * h_v + W_neigh * Mean_{u in N(v)}(h_u))
    """

    def __init__(self, in_features: int, out_features: int):
        super(GraphSAGELayer, self).__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neigh_linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        out_self = self.self_linear(x)

        if edge_index.numel() == 0:
            return F.relu(out_self)

        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(N, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = torch.clamp(deg, min=1.0)

        neigh_x = x[src]
        aggregated = torch.zeros((N, x.size(1)), device=x.device, dtype=x.dtype)
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(neigh_x), neigh_x)
        aggregated = aggregated / deg.unsqueeze(1)

        out_neigh = self.neigh_linear(aggregated)
        return F.relu(out_self + out_neigh)


class GraphSAGEModel(nn.Module):
    """Standard Two-layer GraphSAGE network."""

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = GraphSAGELayer(in_features, hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.dropout(h)
        return self.classifier(h).squeeze(-1)


class DeepResGraphSAGEModel(nn.Module):
    """
    Deep 3-Layer GraphSAGE with Residual Skip Connections and Layer Normalization.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super(DeepResGraphSAGEModel, self).__init__()
        self.conv1 = GraphSAGELayer(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv3 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = self.dropout(h1)

        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(h2 + h1)  # Residual skip connection
        h2 = self.dropout(h2)

        h3 = self.conv3(h2, edge_index)
        h3 = self.norm3(h3 + h2)  # Residual skip connection
        h3 = self.dropout(h3)

        return self.classifier(h3).squeeze(-1)


class GCNLayer(nn.Module):
    r"""
    Graph Convolutional Network (GCN) Layer in PyTorch.
    h_v = ReLU( \sum_{u \in N(v) \cup {v}} (d_u d_v)^{-1/2} W h_u )
    """

    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.linear(x)

        if edge_index.numel() == 0:
            return F.relu(h)

        src, dst = edge_index[0], edge_index[1]
        self_loops = torch.arange(N, device=x.device).unsqueeze(0).repeat(2, 1)
        full_edges = torch.cat([edge_index, self_loops], dim=1)

        s, d = full_edges[0], full_edges[1]
        deg = torch.zeros(N, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, d, torch.ones_like(d, dtype=x.dtype))
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1.0), -0.5)

        norm = deg_inv_sqrt[s] * deg_inv_sqrt[d]
        norm_h = h[s] * norm.unsqueeze(1)

        out = torch.zeros((N, h.size(1)), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, d.unsqueeze(1).expand_as(norm_h), norm_h)
        return F.relu(out)


class GCNModel(nn.Module):
    """Two-layer GCN Model."""

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, edge_index)
        h = self.dropout(h)
        h = self.gcn2(h, edge_index)
        h = self.dropout(h)
        return self.classifier(h).squeeze(-1)


class MLPGraphModel(nn.Module):
    """Tabular MLP Baseline ignoring graph connections."""

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super(MLPGraphModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.classifier(h).squeeze(-1)


class EllipticGNNPipeline:
    """
    Manager for data preparation, temporal splits, model training, TorchScript export, and benchmarking.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        lr: float = 0.005,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        classes_df: pd.DataFrame,
        edgelist_df: pd.DataFrame
    ) -> Dict[str, Any]:
        df_merged = pd.merge(features_df, classes_df[['txId', 'class_mapped']], on='txId', how='left')

        unique_txs = df_merged['txId'].unique()
        tx2idx = {tx: idx for idx, tx in enumerate(unique_txs)}

        feat_cols = [c for c in features_df.columns if c not in ['txId', 'time_step']]
        X_mat = df_merged[feat_cols].values.astype(np.float32)

        mean = np.nanmean(X_mat, axis=0)
        std = np.nanstd(X_mat, axis=0) + 1e-5
        X_mat = np.nan_to_num((X_mat - mean) / std)

        y_vec = df_merged['class_mapped'].values.astype(np.int64)
        time_steps = df_merged['time_step'].values

        valid_edges = edgelist_df[
            edgelist_df['txId1'].isin(tx2idx) & edgelist_df['txId2'].isin(tx2idx)
        ]
        src = [tx2idx[tx] for tx in valid_edges['txId1']]
        dst = [tx2idx[tx] for tx in valid_edges['txId2']]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        train_mask = (time_steps <= 34) & (y_vec != -1)
        val_mask = (time_steps >= 35) & (time_steps <= 39) & (y_vec != -1)
        test_mask = (time_steps >= 40) & (y_vec != -1)

        return {
            "x": torch.tensor(X_mat, dtype=torch.float32),
            "y": torch.tensor(y_vec, dtype=torch.float32),
            "edge_index": edge_index,
            "train_mask": torch.tensor(train_mask, dtype=torch.bool),
            "val_mask": torch.tensor(val_mask, dtype=torch.bool),
            "test_mask": torch.tensor(test_mask, dtype=torch.bool),
            "tx2idx": tx2idx,
            "num_features": X_mat.shape[1]
        }

    def train_model(
        self,
        model: nn.Module,
        data: Dict[str, Any],
        epochs: int = 30,
        verbose: bool = False
    ) -> Dict[str, Any]:
        x = data["x"].to(self.device)
        y = data["y"].to(self.device)
        edge_index = data["edge_index"].to(self.device)
        train_mask = data["train_mask"].to(self.device)

        y_train = y[train_mask]
        num_pos = (y_train == 1).sum().item()
        num_neg = (y_train == 0).sum().item()
        pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = criterion(logits[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

        train_time = time.time() - t0
        test_metrics = self.evaluate_model(model, data, mask_name="test_mask")
        test_metrics["train_time"] = round(train_time, 3)
        return test_metrics

    def train(self, data: Dict[str, Any], epochs: int = 30, verbose: bool = True) -> Dict[str, Any]:
        self.model = GraphSAGEModel(
            in_features=data["num_features"],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)

        return self.train_model(self.model, data, epochs=epochs, verbose=verbose)

    def evaluate_model(self, model: nn.Module, data: Dict[str, Any], mask_name: str = "test_mask") -> Dict[str, Any]:
        model.eval()
        x = data["x"].to(self.device)
        y = data["y"].to(self.device)
        edge_index = data["edge_index"].to(self.device)
        mask = data[mask_name].to(self.device)

        t_inf = time.time()
        with torch.no_grad():
            logits = model(x, edge_index)
            probs = torch.sigmoid(logits)[mask].cpu().numpy()
            y_true = y[mask].cpu().numpy().astype(int)

        num_eval = int(mask.sum().item())
        inf_latency_ms = ((time.time() - t_inf) / max(num_eval, 1)) * 1000.0

        preds = (probs >= 0.5).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5

        p_curve, r_curve, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(r_curve, p_curve)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "latency_ms_per_sample": round(inf_latency_ms, 4),
            "sample_count": len(y_true),
            "illicit_count": int(y_true.sum())
        }

    def evaluate(self, data: Dict[str, Any], mask_name: str = "test_mask") -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model must be trained first.")
        return self.evaluate_model(self.model, data, mask_name=mask_name)

    def benchmark_models(self, data: Dict[str, Any], epochs: int = 20) -> pd.DataFrame:
        """
        Runs multi-model GNN spectrum comparison on temporal test split (small to deep):
        - MLP Baseline (Small, no graph structure)
        - GCN (Medium Graph Convolution)
        - GraphSAGE (Large Neighborhood Aggregation)
        - Deep ResGraphSAGE (Deep 3-Layer GNN with Residual Skip Connections)
        """
        in_feats = data["num_features"]

        candidates = {
            "MLP Baseline": (MLPGraphModel(in_feats, hidden_dim=self.hidden_dim).to(self.device), "Small"),
            "GCN": (GCNModel(in_feats, hidden_dim=self.hidden_dim).to(self.device), "Medium"),
            "GraphSAGE": (GraphSAGEModel(in_feats, hidden_dim=self.hidden_dim).to(self.device), "Large"),
            "Deep ResGraphSAGE": (DeepResGraphSAGEModel(in_feats, hidden_dim=self.hidden_dim).to(self.device), "Deep")
        }

        results = []

        for name, (model, tier) in candidates.items():
            metrics = self.train_model(model, data, epochs=epochs)
            results.append({
                "Model": name,
                "Model Tier": tier,
                "Precision": float(metrics["precision"]),
                "Recall": float(metrics["recall"]),
                "F1 Score": float(metrics["f1"]),
                "PR-AUC": float(metrics["pr_auc"]),
                "ROC-AUC": float(metrics["roc_auc"]),
                "Latency (ms/sample)": metrics["latency_ms_per_sample"],
                "Train Time (s)": metrics["train_time"]
            })

        return pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)

    def export_torchscript(self, export_dir: str, in_features: int) -> str:
        """
        Exports PyTorch model via TorchScript tracing for high-performance C++/Triton serving.
        """
        os.makedirs(export_dir, exist_ok=True)

        if self.model is None:
            self.model = GraphSAGEModel(in_features=in_features, hidden_dim=self.hidden_dim).to(self.device)

        self.model.eval()

        dummy_x = torch.randn(10, in_features).to(self.device)
        dummy_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).to(self.device)

        traced_script_module = torch.jit.trace(self.model, (dummy_x, dummy_edge_index))
        export_path = os.path.join(export_dir, "elliptic_graphsage.ptc")
        traced_script_module.save(export_path)

        state_dict_path = os.path.join(export_dir, "elliptic_graphsage.pt")
        torch.save(self.model.state_dict(), state_dict_path)

        print(f"[EllipticGNNPipeline] TorchScript model exported to {export_path}")
        return export_path
