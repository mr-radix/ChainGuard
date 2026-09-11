import pytest
import os
import pandas as pd
from src.data_loading import load_bitcoinheist, load_elliptic


def test_load_bitcoinheist():
    if not os.path.exists("Dataset/Bitcoin Heist Ransomware Address/BitcoinHeistData.csv"):
        pytest.skip("Dataset file not available locally")

    df = load_bitcoinheist(sample_size=100)
    assert len(df) == 100
    assert 'is_ransomware' in df.columns
    assert 'income' in df.columns
    assert 'label' in df.columns


def test_load_elliptic():
    if not os.path.exists("Dataset/Elliptic/elliptic_txs_classes.csv"):
        pytest.skip("Elliptic dataset not available locally")

    features_df, classes_df, edgelist_df = load_elliptic(sample_size=100)
    assert len(features_df) <= 100
    assert 'class_mapped' in classes_df.columns
    assert 'txId1' in edgelist_df.columns or len(edgelist_df) == 0
