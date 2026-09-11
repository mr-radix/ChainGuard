import pandas as pd
from src.graph_analysis import (
    build_flow_graph, trace_fund_flow, detect_peel_chain, 
    detect_fan_out_fan_in, common_input_clustering
)


def test_union_find_clustering():
    tx_inputs = pd.DataFrame({
        'tx_hash': ['tx1', 'tx1', 'tx2', 'tx2'],
        'address': ['addr1', 'addr2', 'addr2', 'addr3']
    })
    clusters = common_input_clustering(tx_inputs)
    assert clusters['addr1'] == clusters['addr2']
    assert clusters['addr2'] == clusters['addr3']


def test_build_and_trace_graph():
    tx_df = pd.DataFrame({
        'from_address': ['A', 'B', 'C'],
        'to_address': ['B', 'C', 'D'],
        'amount': [10.0, 9.0, 8.0],
        'timestamp': [100, 200, 300],
        'chain': ['BTC', 'BTC', 'BTC'],
        'tx_hash': ['tx1', 'tx2', 'tx3']
    })

    G = build_flow_graph(tx_df)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3

    traces = trace_fund_flow(G, seed_address='A', min_amount=1.0, max_depth=3)
    assert len(traces) == 3
    assert traces[0]['from_address'] == 'A'
    assert traces[0]['to_address'] == 'B'


def test_detect_peel_chain():
    # Build synthetic peel chain A -> B (9.0 main, 1.0 peel), B -> C (8.0 main, 1.0 peel), C -> D (7.0 main, 1.0 peel)
    tx_df = pd.DataFrame({
        'from_address': ['A', 'A', 'B', 'B', 'C', 'C'],
        'to_address': ['B', 'X1', 'C', 'X2', 'D', 'X3'],
        'amount': [9.0, 1.0, 8.0, 1.0, 7.0, 1.0],
        'timestamp': [10, 10, 20, 20, 30, 30],
        'chain': ['BTC']*6,
        'tx_hash': [f'tx{i}' for i in range(6)]
    })

    G = build_flow_graph(tx_df)
    chains = detect_peel_chain(G, start_node='A', min_length=3, max_peel_ratio=0.35)
    assert len(chains) == 1
    assert chains[0]['length'] == 3
    assert chains[0]['total_peeled_amount'] == 3.0


def test_fan_out_fan_in():
    tx_df = pd.DataFrame({
        'from_address': ['Source']*5 + ['A', 'B', 'C', 'D', 'E'],
        'to_address': ['D1', 'D2', 'D3', 'D4', 'D5'] + ['Sink']*5,
        'amount': [1.0]*10,
        'timestamp': [100]*10,
        'chain': ['BTC']*10,
        'tx_hash': [f'tx{i}' for i in range(10)]
    })
    G = build_flow_graph(tx_df)
    results = detect_fan_out_fan_in(G, min_fan_degree=4)

    assert len(results['fan_out']) >= 1
    assert results['fan_out'][0]['address'] == 'Source'

    assert len(results['fan_in']) >= 1
    assert results['fan_in'][0]['address'] == 'Sink'
