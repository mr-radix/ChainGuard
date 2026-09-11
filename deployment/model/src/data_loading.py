import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union

BITCOIN_HEIST_DEFAULT_PATHS = [
    "Dataset/Bitcoin Heist Ransomware Address/BitcoinHeistData.csv",
    "../Dataset/Bitcoin Heist Ransomware Address/BitcoinHeistData.csv",
    "data/bitcoinheist/BitcoinHeistData.csv",
    "../data/bitcoinheist/BitcoinHeistData.csv",
    "BitcoinHeistData.csv"
]

ELLIPTIC_DEFAULT_DIRS = [
    "Dataset/Elliptic",
    "../Dataset/Elliptic",
    "data/elliptic",
    "../data/elliptic",
    "Elliptic",
    "../Elliptic"
]


def load_bitcoinheist(
    path_or_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads the BitcoinHeist Ransomware dataset.
    Columns expected: ['address', 'year', 'day', 'length', 'weight', 'count', 
                       'looped', 'neighbors', 'income', 'label']
    """
    resolved_path = None
    if path_or_dir and os.path.isfile(path_or_dir):
        resolved_path = path_or_dir
    elif path_or_dir and os.path.isdir(path_or_dir):
        candidate = os.path.join(path_or_dir, "BitcoinHeistData.csv")
        if os.path.exists(candidate):
            resolved_path = candidate
        else:
            candidate2 = os.path.join(path_or_dir, "Bitcoin Heist Ransomware Address", "BitcoinHeistData.csv")
            if os.path.exists(candidate2):
                resolved_path = candidate2

    if not resolved_path:
        for default_p in BITCOIN_HEIST_DEFAULT_PATHS:
            if os.path.exists(default_p):
                resolved_path = default_p
                break

    if not resolved_path:
        raise FileNotFoundError(
            f"BitcoinHeist dataset not found. Checked default locations: {BITCOIN_HEIST_DEFAULT_PATHS}"
        )

    print(f"[data_loading] Loading BitcoinHeist dataset from {resolved_path}...")
    if sample_size:
        df = pd.read_csv(resolved_path)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        df = pd.read_csv(resolved_path)

    # Ensure correct data types
    numeric_cols = ['year', 'day', 'length', 'weight', 'count', 'looped', 'neighbors', 'income']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Create binary target column
    df['is_ransomware'] = (df['label'].astype(str).str.lower() != 'white').astype(int)

    print(f"[data_loading] BitcoinHeist loaded: {len(df):,} rows. Ransomware samples: {df['is_ransomware'].sum():,}")
    return df


def load_elliptic(
    data_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads Elliptic dataset components:
    - Features: txId, time_step, feature_1 .. feature_165
    - Classes: txId, class (1: illicit, 0: licit, -1: unknown)
    - Edgelist: txId1, txId2
    """
    resolved_dir = None
    if data_dir and os.path.isdir(data_dir):
        resolved_dir = data_dir
        if not os.path.exists(os.path.join(resolved_dir, "elliptic_txs_classes.csv")):
            sub = os.path.join(resolved_dir, "Elliptic")
            if os.path.exists(os.path.join(sub, "elliptic_txs_classes.csv")):
                resolved_dir = sub

    if not resolved_dir:
        for default_d in ELLIPTIC_DEFAULT_DIRS:
            if os.path.exists(os.path.join(default_d, "elliptic_txs_classes.csv")):
                resolved_dir = default_d
                break

    if not resolved_dir:
        raise FileNotFoundError(
            f"Elliptic dataset not found. Checked default directories: {ELLIPTIC_DEFAULT_DIRS}"
        )

    classes_path = os.path.join(resolved_dir, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(resolved_dir, "elliptic_txs_edgelist.csv")
    features_path = os.path.join(resolved_dir, "elliptic_txs_features.csv")

    print(f"[data_loading] Loading Elliptic dataset from {resolved_dir}...")
    
    # 1. Load classes
    classes_df = pd.read_csv(classes_path)
    class_map = {'1': 1, '2': 0, 'unknown': -1, 1: 1, 2: 0}
    classes_df['class_mapped'] = classes_df['class'].map(lambda x: class_map.get(str(x), -1))

    # 2. Load edgelist
    edgelist_df = pd.read_csv(edgelist_path)

    # 3. Load features (no header in original CSV)
    features_df = pd.read_csv(features_path, header=None)
    feature_cols = ['txId', 'time_step'] + [f'feature_{i}' for i in range(1, features_df.shape[1] - 1)]
    features_df.columns = feature_cols

    if sample_size and len(features_df) > sample_size:
        print(f"[data_loading] Sampling Elliptic dataset to {sample_size:,} transactions...")
        sampled_tx_ids = set(features_df['txId'].sample(n=sample_size, random_state=random_state))
        features_df = features_df[features_df['txId'].isin(sampled_tx_ids)].reset_index(drop=True)
        classes_df = classes_df[classes_df['txId'].isin(sampled_tx_ids)].reset_index(drop=True)
        edgelist_df = edgelist_df[
            edgelist_df['txId1'].isin(sampled_tx_ids) & edgelist_df['txId2'].isin(sampled_tx_ids)
        ].reset_index(drop=True)

    print(
        f"[data_loading] Elliptic loaded: {len(features_df):,} transactions, "
        f"{len(edgelist_df):,} edges. Illicit: {(classes_df['class_mapped'] == 1).sum():,}, "
        f"Licit: {(classes_df['class_mapped'] == 0).sum():,}, "
        f"Unknown: {(classes_df['class_mapped'] == -1).sum():,}"
    )

    return features_df, classes_df, edgelist_df
