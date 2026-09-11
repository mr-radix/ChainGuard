import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Set, Tuple, Optional


class UnionFind:
    """Disjoint Set Union (DSU) data structure for UTXO address clustering."""

    def __init__(self):
        self.parent = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root2] = root1

    def get_clusters(self) -> Dict[str, str]:
        return {item: self.find(item) for item in self.parent}


def common_input_clustering(tx_inputs_df: pd.DataFrame) -> Dict[str, str]:
    """
    UTXO Common-Input Ownership Heuristic.
    Addresses co-signing inputs of the same transaction belong to the same entity.
    tx_inputs_df columns: [tx_hash, address]
    Returns mapping: address -> cluster_id
    """
    uf = UnionFind()

    grouped = tx_inputs_df.groupby('tx_hash')['address'].apply(list)
    for addresses in grouped:
        if len(addresses) > 1:
            first_addr = addresses[0]
            for addr in addresses[1:]:
                uf.union(first_addr, addr)

    return uf.get_clusters()


def build_flow_graph(tx_df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Builds a directed multigraph from normalized transaction records.
    tx_df columns: [from_address, to_address, amount, timestamp, chain, tx_hash]
    """
    G = nx.MultiDiGraph()

    for _, row in tx_df.iterrows():
        from_addr = str(row['from_address'])
        to_addr = str(row['to_address'])
        amount = float(row.get('amount', 0.0))
        timestamp = float(row.get('timestamp', 0.0))
        chain = str(row.get('chain', 'BTC'))
        tx_hash = str(row.get('tx_hash', ''))

        G.add_node(from_addr, type='address')
        G.add_node(to_addr, type='address')
        G.add_edge(
            from_addr,
            to_addr,
            amount=amount,
            timestamp=timestamp,
            chain=chain,
            tx_hash=tx_hash
        )

    return G


def trace_fund_flow(
    graph: nx.MultiDiGraph,
    seed_address: str,
    min_amount: float = 0.0,
    max_depth: int = 4
) -> List[Dict[str, Any]]:
    """
    BFS/DFS traversal tracing outgoing fund flows from seed_address up to max_depth.
    """
    if seed_address not in graph:
        return []

    traces = []
    queue = [(seed_address, 0, [seed_address], 0.0)]

    while queue:
        curr_node, depth, path, cumulative_amount = queue.pop(0)

        if depth >= max_depth:
            continue

        out_edges = graph.out_edges(curr_node, data=True)
        for u, v, data in out_edges:
            amt = data.get('amount', 0.0)
            if amt < min_amount:
                continue

            new_cumulative = cumulative_amount + amt
            new_path = path + [v]

            trace_item = {
                "depth": depth + 1,
                "from_address": u,
                "to_address": v,
                "amount": amt,
                "cumulative_amount": new_cumulative,
                "timestamp": data.get('timestamp', 0.0),
                "chain": data.get('chain', 'BTC'),
                "tx_hash": data.get('tx_hash', ''),
                "path": new_path
            }
            traces.append(trace_item)

            if v not in path:  # Prevent infinite loops in cycles
                queue.append((v, depth + 1, new_path, new_cumulative))

    return traces


def detect_peel_chain(
    graph: nx.MultiDiGraph,
    start_node: str,
    min_length: int = 3,
    max_peel_ratio: float = 0.35
) -> List[Dict[str, Any]]:
    """
    Detects peel-chain patterns: sequential hops where one primary destination receives 
    the bulk of funds and a smaller peeled portion goes elsewhere.
    """
    peel_chains = []
    if start_node not in graph:
        return peel_chains

    curr = start_node
    chain_hops = []

    while True:
        out_edges = list(graph.out_edges(curr, data=True))
        if len(out_edges) < 2:
            break

        total_out = sum(d.get('amount', 0.0) for _, _, d in out_edges)
        if total_out <= 0:
            break

        # Sort edges by amount descending
        sorted_edges = sorted(out_edges, key=lambda e: e[2].get('amount', 0.0), reverse=True)
        main_edge = sorted_edges[0]
        peel_edge = sorted_edges[1]

        main_amt = main_edge[2].get('amount', 0.0)
        peel_amt = peel_edge[2].get('amount', 0.0)
        peel_ratio = peel_amt / total_out

        if peel_ratio <= max_peel_ratio:
            next_hop = main_edge[1]
            peel_dest = peel_edge[1]
            chain_hops.append({
                "from_address": curr,
                "main_pass_to": next_hop,
                "main_amount": main_amt,
                "peeled_to": peel_dest,
                "peeled_amount": peel_amt,
                "peel_ratio": peel_ratio
            })
            curr = next_hop
            if curr in [h["from_address"] for h in chain_hops]:
                break  # Cycle detected
        else:
            break

    if len(chain_hops) >= min_length:
        peel_chains.append({
            "start_address": start_node,
            "length": len(chain_hops),
            "hops": chain_hops,
            "total_peeled_amount": sum(h["peeled_amount"] for h in chain_hops)
        })

    return peel_chains


def detect_fan_out_fan_in(
    graph: nx.MultiDiGraph,
    min_fan_degree: int = 4
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identifies high out-degree Fan-Out nodes (dispersion) and high in-degree Fan-In nodes (consolidation).
    """
    fan_out_nodes = []
    fan_in_nodes = []

    for node in graph.nodes():
        out_degree = graph.out_degree(node)
        in_degree = graph.in_degree(node)

        if out_degree >= min_fan_degree and out_degree > 2 * max(in_degree, 1):
            out_edges = list(graph.out_edges(node, data=True))
            total_dispensed = sum(d.get('amount', 0.0) for _, _, d in out_edges)
            fan_out_nodes.append({
                "address": node,
                "out_degree": out_degree,
                "in_degree": in_degree,
                "total_dispensed_amount": total_dispensed,
                "fan_ratio": out_degree / max(in_degree, 1)
            })

        if in_degree >= min_fan_degree and in_degree > 2 * max(out_degree, 1):
            in_edges = list(graph.in_edges(node, data=True))
            total_consolidated = sum(d.get('amount', 0.0) for _, _, d in in_edges)
            fan_in_nodes.append({
                "address": node,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_consolidated_amount": total_consolidated,
                "fan_ratio": in_degree / max(out_degree, 1)
            })

    return {
        "fan_out": sorted(fan_out_nodes, key=lambda x: x["out_degree"], reverse=True),
        "fan_in": sorted(fan_in_nodes, key=lambda x: x["in_degree"], reverse=True)
    }
