import pandas as pd
import numpy as np
from src.features import bitcoinheist_features, address_behavior_profile


def test_bitcoinheist_features():
    df = pd.DataFrame({
        'address': ['addr1', 'addr2'],
        'year': [2017, 2018],
        'day': [100, 150],
        'length': [10, 20],
        'weight': [0.5, 1.2],
        'count': [5, 10],
        'looped': [1, 2],
        'neighbors': [3, 4],
        'income': [1000.0, 5000.0],
        'label': ['white', 'princetonLocky']
    })

    df_feat = bitcoinheist_features(df)
    assert 'income_per_neighbor' in df_feat.columns
    assert 'log_income' in df_feat.columns
    assert 'temporal_day' in df_feat.columns
    assert df_feat['log_income'].iloc[0] > 0


def test_address_behavior_profile():
    tx_df = pd.DataFrame({
        'chain': ['BTC', 'BTC', 'BTC'],
        'tx_hash': ['tx1', 'tx2', 'tx3'],
        'from_address': ['addrA', 'addrA', 'addrB'],
        'to_address': ['addrB', 'addrC', 'addrC'],
        'amount': [10.0, 5.0, 12.0],
        'timestamp': [1000, 2000, 3000]
    })

    profile = address_behavior_profile(tx_df)
    assert not profile.empty
    assert 'addrA' in profile['address'].values
    assert 'fan_out_ratio' in profile.columns
    assert profile.loc[profile['address'] == 'addrA', 'sent_count'].values[0] == 2
