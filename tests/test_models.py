import pandas as pd
import numpy as np
import torch
from src.models.ransomware_model import RansomwareClassifier, RansomwareModelSuite
from src.models.elliptic_gnn import EllipticGNNPipeline, GraphSAGEModel, GCNModel, MLPGraphModel


def test_ransomware_classifier():
    df = pd.DataFrame({
        'address': [f'addr_{i}' for i in range(100)],
        'year': [2017]*100,
        'day': list(range(100)),
        'length': np.random.randint(1, 50, 100),
        'weight': np.random.uniform(0.1, 10.0, 100),
        'count': np.random.randint(1, 20, 100),
        'looped': np.random.randint(0, 5, 100),
        'neighbors': np.random.randint(1, 10, 100),
        'income': np.random.uniform(100, 10000, 100),
        'label': ['white']*80 + ['princetonLocky']*10 + ['princetonCerber']*10,
        'is_ransomware': [0]*80 + [1]*20
    })

    model = RansomwareClassifier()
    metrics = model.fit(df)
    assert metrics['precision'] >= 0.0
    assert metrics['recall'] >= 0.0
    assert metrics['f1'] >= 0.0

    scores = model.predict_risk_score(df[:10])
    assert len(scores) == 10
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_ransomware_model_suite():
    df = pd.DataFrame({
        'address': [f'addr_{i}' for i in range(100)],
        'year': [2017]*100,
        'day': list(range(100)),
        'length': np.random.randint(1, 50, 100),
        'weight': np.random.uniform(0.1, 10.0, 100),
        'count': np.random.randint(1, 20, 100),
        'looped': np.random.randint(0, 5, 100),
        'neighbors': np.random.randint(1, 10, 100),
        'income': np.random.uniform(100, 10000, 100),
        'label': ['white']*80 + ['princetonLocky']*10 + ['princetonCerber']*10,
        'is_ransomware': [0]*80 + [1]*20
    })

    suite = RansomwareModelSuite()
    benchmarks = suite.train_and_benchmark(df)
    assert not benchmarks.empty
    assert 'Model' in benchmarks.columns
    assert 'PR-AUC' in benchmarks.columns
    assert len(benchmarks) == 5


def test_elliptic_gnn_model_suite():
    model_sage = GraphSAGEModel(in_features=165, hidden_dim=32)
    model_gcn = GCNModel(in_features=165, hidden_dim=32)
    model_mlp = MLPGraphModel(in_features=165, hidden_dim=32)

    x = torch.randn(20, 165)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    out1 = model_sage(x, edge_index)
    out2 = model_gcn(x, edge_index)
    out3 = model_mlp(x, edge_index)

    assert out1.shape == (20,)
    assert out2.shape == (20,)
    assert out3.shape == (20,)
