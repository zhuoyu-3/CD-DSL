from typing import List, Set

import numpy as np


def graph_neighbors(A: np.ndarray, node: int) -> List[int]:
    return [dst for dst in range(A.shape[1]) if node != dst and A[node, dst] > 0]


def reverse_graph_neighbors(A: np.ndarray, node: int) -> List[int]:
    return [src for src in range(A.shape[0]) if node != src and A[src, node] > 0]


def reachable_nodes(A: np.ndarray, start: int, reverse: bool = False) -> Set[int]:
    visit = reverse_graph_neighbors if reverse else graph_neighbors
    seen: Set[int] = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in visit(A, node):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def is_strongly_connected(A: np.ndarray) -> bool:
    clients = A.shape[0]
    if clients <= 1:
        return True
    return (
        len(reachable_nodes(A, 0, reverse=False)) == clients
        and len(reachable_nodes(A, 0, reverse=True)) == clients
    )


def minimum_connection_rate(clients: int) -> float:
    if clients <= 1:
        return 0.0
    return float(min(1.0, max(1.0 / (clients - 1), 1.5 * np.log(clients) / clients)))


def build_adjacency(
    clients: int,
    topology: str,
    extra_edges: int,
    graph_degree: int,
    connection_rate: float,
    topology_max_retries: int,
) -> np.ndarray:
    topology = topology.lower()
    max_retries = max(1, int(topology_max_retries))
    random_topology = topology in {"random_degree", "random_rate"}
    effective_rate = float(min(1.0, max(0.0, connection_rate)))
    if topology == "random_rate":
        effective_rate = max(effective_rate, minimum_connection_rate(clients))

    for _attempt in range(max_retries):
        A = np.eye(clients, dtype=np.float64)
        if topology == "directed_ring":
            for src in range(clients):
                A[src, (src + 1) % clients] = 1.0
                for hop in range(2, extra_edges + 2):
                    A[src, (src + hop) % clients] = 1.0
        elif topology == "bidirectional_ring":
            for src in range(clients):
                for hop in range(1, extra_edges + 2):
                    A[src, (src + hop) % clients] = 1.0
                    A[src, (src - hop) % clients] = 1.0
        elif topology == "fully_connected":
            A[:] = 1.0
        elif topology == "random_degree":
            degree = max(0, min(int(graph_degree), clients - 1))
            for src in range(clients):
                chosen = set()
                if degree > 0:
                    ring_dst = (src + 1) % clients
                    chosen.add(ring_dst)
                    A[src, ring_dst] = 1.0

                remaining = degree - len(chosen)
                if remaining <= 0:
                    continue

                candidates = [
                    dst for dst in range(clients)
                    if dst != src and dst not in chosen
                ]
                sampled = np.random.choice(candidates, size=remaining, replace=False)
                for dst in sampled:
                    A[src, int(dst)] = 1.0
        elif topology == "random_rate":
            for src in range(clients):
                for dst in range(clients):
                    if dst == src:
                        continue
                    if np.random.rand() < effective_rate:
                        A[src, dst] = 1.0
        else:
            raise ValueError(f"Unsupported topology: {topology}")

        if is_strongly_connected(A):
            return A
        if not random_topology:
            break

    if topology == "random_rate":
        raise RuntimeError(
            "Failed to sample a strongly connected random_rate graph after "
            f"{max_retries} retries. Increase --connection-rate above "
            f"{minimum_connection_rate(clients):.4f} or raise --topology-max-retries."
        )
    if topology == "random_degree":
        raise RuntimeError(
            "Failed to sample a strongly connected random_degree graph after "
            f"{max_retries} retries. Increase --graph-degree or raise "
            "--topology-max-retries."
        )
    raise RuntimeError(f"Topology {topology} is not strongly connected.")


def column_stochastic_mixing(A: np.ndarray) -> np.ndarray:
    P = A.astype(np.float64).copy()
    np.fill_diagonal(P, 1.0)
    col_sums = P.sum(axis=0, keepdims=True)
    if np.any(col_sums == 0):
        raise ValueError("Each node must have at least one in-neighbor.")
    return P / col_sums


def incoming_neighbors(P: np.ndarray, node: int) -> List[int]:
    return [src for src in range(P.shape[0]) if P[src, node] > 0]


def neighbor_support(A: np.ndarray, node: int) -> List[int]:
    return [src for src in range(A.shape[0]) if A[src, node] > 0]
