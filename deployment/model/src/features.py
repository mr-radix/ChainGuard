import pandas as pd
import numpy as np
from typing import Optional

DEFAULT_BH_COLS = ['income', 'neighbors', 'count', 'looped', 'weight', 'length', 'year', 'day']


def bitcoinheist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers behavioral, ratio, and log features for BitcoinHeist dataset.
    Fills any missing expected columns with default zeros.
    """
    df = df.copy()

    for col in DEFAULT_BH_COLS:
        if col not in df.columns:
            df[col] = 0.0

    eps = 1e-5

    # Ratios
    df['income_per_neighbor'] = df['income'] / (df['neighbors'] + eps)
    df['income_per_count'] = df['income'] / (df['count'] + eps)
    df['loop_ratio'] = df['looped'] / (df['count'] + eps)
    df['weight_per_count'] = df['weight'] / (df['count'] + eps)
    df['length_per_count'] = df['length'] / (df['count'] + eps)

    # Log transformations for skewed columns
    df['log_income'] = np.log1p(np.maximum(0, df['income']))
    df['log_weight'] = np.log1p(np.maximum(0, df['weight']))
    df['log_count'] = np.log1p(np.maximum(0, df['count']))
    df['log_neighbors'] = np.log1p(np.maximum(0, df['neighbors']))

    # Temporal feature
    if 'year' in df.columns and 'day' in df.columns:
        min_year = df['year'].min() if len(df) > 0 and df['year'].min() > 0 else 2009
        df['temporal_day'] = (df['year'] - min_year) * 365 + df['day']

    return df


def elliptic_derived_features(
    features_df: pd.DataFrame,
    edgelist_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    df = features_df.copy()

    if edgelist_df is not None and not edgelist_df.empty:
        out_degrees = edgelist_df.groupby('txId1').size().rename('graph_out_degree')
        in_degrees = edgelist_df.groupby('txId2').size().rename('graph_in_degree')

        df = df.merge(out_degrees, left_on='txId', right_index=True, how='left')
        df = df.merge(in_degrees, left_on='txId', right_index=True, how='left')

        df['graph_out_degree'] = df['graph_out_degree'].fillna(0)
        df['graph_in_degree'] = df['graph_in_degree'].fillna(0)
        df['graph_total_degree'] = df['graph_out_degree'] + df['graph_in_degree']
        df['graph_degree_ratio'] = (df['graph_out_degree'] + 1e-5) / (df['graph_in_degree'] + 1e-5)

    return df


def address_behavior_profile(tx_df: pd.DataFrame) -> pd.DataFrame:
    if tx_df.empty:
        return pd.DataFrame()

    sent_df = tx_df.groupby('from_address').agg(
        sent_count=('tx_hash', 'count'),
        sent_amount_sum=('amount', 'sum'),
        sent_amount_mean=('amount', 'mean'),
        sent_amount_std=('amount', 'std'),
        unique_destinations=('to_address', 'nunique'),
        min_sent_time=('timestamp', 'min'),
        max_sent_time=('timestamp', 'max')
    ).reset_index().rename(columns={'from_address': 'address'})

    recv_df = tx_df.groupby('to_address').agg(
        received_count=('tx_hash', 'count'),
        received_amount_sum=('amount', 'sum'),
        received_amount_mean=('amount', 'mean'),
        received_amount_std=('amount', 'std'),
        unique_sources=('from_address', 'nunique'),
        min_recv_time=('timestamp', 'min'),
        max_recv_time=('timestamp', 'max')
    ).reset_index().rename(columns={'to_address': 'address'})

    profile = pd.merge(sent_df, recv_df, on='address', how='outer').fillna(0)

    profile['tx_count'] = profile['sent_count'] + profile['received_count']
    profile['total_volume'] = profile['sent_amount_sum'] + profile['received_amount_sum']
    profile['net_flow'] = profile['received_amount_sum'] - profile['sent_amount_sum']

    eps = 1e-5
    profile['fan_out_ratio'] = profile['unique_destinations'] / (profile['sent_count'] + eps)
    profile['fan_in_ratio'] = profile['unique_sources'] / (profile['received_count'] + eps)
    profile['out_in_count_ratio'] = profile['sent_count'] / (profile['received_count'] + eps)

    min_time = np.minimum(
        profile['min_sent_time'].replace(0, np.nan),
        profile['min_recv_time'].replace(0, np.nan)
    ).fillna(0)
    max_time = np.maximum(profile['max_sent_time'], profile['max_recv_time'])

    profile['active_duration_sec'] = max_time - min_time
    profile['tx_velocity_per_hour'] = profile['tx_count'] / (
        (profile['active_duration_sec'] / 3600.0) + 1.0
    )

    return profile
