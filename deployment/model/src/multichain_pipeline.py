import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any

NORMALIZED_COLUMNS = ['chain', 'tx_hash', 'from_address', 'to_address', 'amount', 'fee', 'timestamp']


def normalize_utxo_chain(raw_df: pd.DataFrame, chain_name: str = "BTC") -> pd.DataFrame:
    """
    Normalizes UTXO chain data (Bitcoin, Litecoin) into standard schema.
    Supported column variants:
    - BitcoinHeist: address, income, year, day
    - Standard UTXO: tx_hash, from_address, to_address, amount, fee, timestamp
    """
    df = raw_df.copy()
    df['chain'] = chain_name.upper()

    if 'tx_hash' not in df.columns:
        if 'address' in df.columns:
            # Synthetic transaction record construction for BitcoinHeist addresses
            df['tx_hash'] = [f"{chain_name.lower()}_tx_{i:07d}" for i in range(len(df))]
            df['from_address'] = df['address']
            df['to_address'] = df['address'].apply(lambda a: f"sink_{hash(a) % 10000:04d}")
            df['amount'] = df.get('income', 0.0)
            df['fee'] = 0.0
            if 'year' in df.columns and 'day' in df.columns:
                df['timestamp'] = (df['year'] - 2009) * 31536000 + df['day'] * 86400
            else:
                df['timestamp'] = 0.0
        else:
            df['tx_hash'] = "unknown"

    for col in NORMALIZED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in ['amount', 'fee', 'timestamp'] else "unknown"

    return df[NORMALIZED_COLUMNS]


def normalize_account_chain(raw_df: pd.DataFrame, chain_name: str = "ETH") -> pd.DataFrame:
    """
    Normalizes Account-based chain data (Ethereum) into standard schema.
    Expected columns: [hash/tx_hash, from/from_address, to/to_address, value/amount, gas_price/fee, time/timestamp]
    """
    df = raw_df.copy()
    df['chain'] = chain_name.upper()

    col_map = {
        'hash': 'tx_hash',
        'from': 'from_address',
        'to': 'to_address',
        'value': 'amount',
        'gas_price': 'fee',
        'time': 'timestamp'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in NORMALIZED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in ['amount', 'fee', 'timestamp'] else "unknown"

    return df[NORMALIZED_COLUMNS]


def merge_chains(chain_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple normalized chain DataFrames into one unified multi-chain DataFrame.
    """
    valid_dfs = [df for df in chain_dfs if not df.empty]
    if not valid_dfs:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    merged = pd.concat(valid_dfs, ignore_index=True)
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    return merged


def cross_chain_entity_resolution(
    address_clusters: Dict[str, str],
    cross_chain_links: pd.DataFrame
) -> Dict[str, str]:
    """
    Maps cross-chain addresses (e.g. BTC address <-> ETH address) to unified Entity IDs 
    using shared exchange deposit addresses or bridge contracts.
    cross_chain_links columns: [btc_address, eth_address, ltc_address, entity_id]
    """
    unified_map = dict(address_clusters)

    for _, row in cross_chain_links.iterrows():
        entity_id = row.get('entity_id', f"entity_{hash(str(row.values)) % 100000:05d}")
        for col in ['btc_address', 'eth_address', 'ltc_address']:
            if col in row and pd.notna(row[col]) and row[col]:
                unified_map[str(row[col])] = entity_id

    return unified_map
