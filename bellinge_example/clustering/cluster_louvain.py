from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    from community import community_louvain
except ImportError:
    community_louvain = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
random.seed(1314)
np.random.seed(1314)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "raw_graph"
OUTPUT_DIR = SCRIPT_DIR / "community_output"


def load_raw_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.load(INPUT_DIR / "edges_all_links.npy")
    edges_weight = np.load(INPUT_DIR / "edges_weight_all_links.npy").reshape(-1)
    edge_types = np.load(INPUT_DIR / "edge_type_all_links.npy").reshape(-1)
    x_combine_final = np.load(INPUT_DIR / "X_combine_final.npy")
    node_names = np.load(INPUT_DIR / "node_names.npy", allow_pickle=True)
    if edges.shape[0] != edges_weight.shape[0]:
        raise RuntimeError("edges_all_links.npy and edges_weight_all_links.npy length mismatch.")
    if edges.shape[0] != edge_types.shape[0]:
        raise RuntimeError("edges_all_links.npy and edge_type_all_links.npy length mismatch.")
    if x_combine_final.shape[1] != 9:
        raise RuntimeError("X_combine_final.npy must have 9 columns.")
    return edges, edges_weight, edge_types, x_combine_final, node_names


def build_graph(edges: np.ndarray, edge_weight: np.ndarray, node_count: int) -> nx.DiGraph:
    edge_index = edges.astype(int) - 1
    if edge_index.min() < 0 or edge_index.max() >= node_count:
        raise RuntimeError("1-based edge indices are out of range.")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))
    for edge_id, (src, dst) in enumerate(edge_index):
        src_id = int(src)
        dst_id = int(dst)
        weight = float(edge_weight[edge_id])
        if graph.has_edge(src_id, dst_id):
            graph[src_id][dst_id]["weight"] += weight
            graph[src_id][dst_id]["raw_edge_count"] += 1
        else:
            graph.add_edge(src_id, dst_id, weight=weight, raw_edge_count=1)
    return graph


def node_positions(x_combine_final: np.ndarray) -> dict[int, tuple[float, float]]:
    coords = x_combine_final[:, [5, 6]]
    return {idx: (float(coords[idx, 0]), float(coords[idx, 1])) for idx in range(coords.shape[0])}


def raw_node_colors(x_combine_final: np.ndarray) -> list[tuple[float, float, float, float]]:
    colors = []
    for area, is_outlet in zip(x_combine_final[:, 0], x_combine_final[:, 4]):
        if is_outlet == 1:
            colors.append((0.33, 0.66, 0.89, 0.85))
        elif area == 0:
            colors.append((0.83, 0.37, 0.33, 0.75))
        else:
            colors.append((0.96, 0.73, 0.48, 0.8))
    return colors


def draw_graph(
    graph: nx.DiGraph,
    pos: dict[int, tuple[float, float]],
    node_colors: list,
    output_path: Path,
    title: str | None = None,
    node_size: int = 80,
) -> None:
    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.25,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        arrowstyle="-|>",
        arrowsize=6,
        edge_color="grey",
        width=0.8,
        alpha=0.65,
        connectionstyle="arc3,rad=0",
    )
    if title:
        plt.title(title)
    plt.gca().set_facecolor("white")
    plt.axis("off")
    plt.savefig(output_path, format="jpg", dpi=600, bbox_inches="tight")
    plt.close()


def run_louvain(graph: nx.DiGraph) -> dict[int, int]:
    undirected_graph = graph.to_undirected()
    if community_louvain is not None:
        return community_louvain.best_partition(undirected_graph, weight="weight", random_state=1314)
    communities = nx.algorithms.community.louvain_communities(undirected_graph, weight="weight", seed=1314)
    partition: dict[int, int] = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            partition[int(node)] = community_id
    return partition


def community_node_colors(partition: dict[int, int], node_count: int) -> list:
    unique_communities = sorted(set(partition.values()))
    cmap = plt.get_cmap("viridis", len(unique_communities))
    community_to_color_id = {community: idx for idx, community in enumerate(unique_communities)}
    return [cmap(community_to_color_id[partition[node]]) for node in range(node_count)]


def community_positions(partition: dict[int, int], x_combine_final: np.ndarray) -> np.ndarray:
    communities = sorted(set(partition.values()))
    positions = np.zeros((len(communities), 2), dtype=float)
    coords = x_combine_final[:, [5, 6]]
    for new_id, community in enumerate(communities):
        nodes = [node for node, node_community in partition.items() if node_community == community]
        positions[new_id, :] = np.mean(coords[nodes], axis=0)
    return positions


def build_community_graph(
    graph: nx.DiGraph,
    partition: dict[int, int],
    x_combine_final: np.ndarray,
):
    communities = sorted(set(partition.values()))
    community_to_new_id = {community: idx for idx, community in enumerate(communities)}
    new_graph = nx.DiGraph()
    new_graph.add_nodes_from(range(len(communities)))
    edge_weights: dict[tuple[int, int], float] = {}
    for src, dst, data in graph.edges(data=True):
        src_community = community_to_new_id[partition[src]]
        dst_community = community_to_new_id[partition[dst]]
        if src_community == dst_community:
            continue
        edge_key = (src_community, dst_community)
        edge_weights[edge_key] = edge_weights.get(edge_key, 0.0) + float(data.get("weight", 1.0))
    for (src_community, dst_community), weight in edge_weights.items():
        new_graph.add_edge(src_community, dst_community, weight=weight)
    x_new = np.zeros((len(communities), x_combine_final.shape[1]), dtype=float)
    for community in communities:
        new_id = community_to_new_id[community]
        nodes = [node for node, node_community in partition.items() if node_community == community]
        x_new[new_id, :] = np.mean(x_combine_final[nodes], axis=0)
    if new_graph.number_of_edges() == 0:
        edge_index_new = np.empty((2, 0), dtype=int)
        edge_weight_new = np.empty((0,), dtype=float)
        community_edges = np.empty((0, 3), dtype=float)
    else:
        edge_rows = [(int(src), int(dst), float(data["weight"])) for src, dst, data in new_graph.edges(data=True)]
        community_edges = np.array(edge_rows, dtype=float)
        edge_index_new = community_edges[:, :2].astype(int).T
        edge_weight_new = community_edges[:, 2].astype(float)
    cluster_assignments = np.array(
        [[partition[node], node] for node in sorted(partition.keys())],
        dtype=int,
    )
    return new_graph, cluster_assignments, community_edges, x_new, edge_index_new, edge_weight_new


def draw_community_graph(new_graph: nx.DiGraph, positions: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(14, 12))
    pos = {node: (float(positions[node, 0]), float(positions[node, 1])) for node in new_graph.nodes()}
    nx.draw_networkx_nodes(new_graph, pos, node_color="lightblue", node_size=500, edgecolors="black")
    nx.draw_networkx_edges(new_graph, pos, arrowstyle="-|>", arrowsize=18, edge_color="grey", width=2)
    nx.draw_networkx_labels(new_graph, pos, font_size=10, font_color="black")
    plt.title("Clustered Graph by Mean Community Coordinates")
    plt.gca().set_facecolor("white")
    plt.axis("off")
    plt.savefig(output_path, format="jpg", dpi=600, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    edges, edges_weight, edge_types, x_combine_final, node_names = load_raw_graph()
    graph = build_graph(edges, edges_weight, x_combine_final.shape[0])
    pos = node_positions(x_combine_final)
    draw_graph(graph, pos, raw_node_colors(x_combine_final), OUTPUT_DIR / "original_graph.jpg", title="Original Graph")
    partition = run_louvain(graph)
    draw_graph(
        graph,
        pos,
        community_node_colors(partition, x_combine_final.shape[0]),
        OUTPUT_DIR / "clustered_original_graph.jpg",
        title="Original Graph with Louvain Communities",
    )
    new_graph, cluster_assignments, community_edges, x_new, edge_index_new, edge_weight_new = build_community_graph(
        graph, partition, x_combine_final
    )
    community_pos = community_positions(partition, x_combine_final)
    draw_community_graph(new_graph, community_pos, OUTPUT_DIR / "clustered_new_graph.jpg")
    np.save(OUTPUT_DIR / "cluster_assignments.npy", cluster_assignments)
    np.save(OUTPUT_DIR / "community_edges.npy", community_edges)
    np.save(OUTPUT_DIR / "community_positions.npy", community_pos)
    np.save(OUTPUT_DIR / "x_new.npy", x_new)
    np.save(OUTPUT_DIR / "edge_index_new.npy", edge_index_new)
    np.save(OUTPUT_DIR / "edge_weight_new.npy", edge_weight_new)
    np.save(OUTPUT_DIR / "node_names.npy", node_names)
    community_sizes = {
        str(community): int(sum(1 for node_community in partition.values() if node_community == community))
        for community in sorted(set(partition.values()))
    }
    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "node_count": int(graph.number_of_nodes()),
        "raw_edge_count": int(edges.shape[0]),
        "edge_type_counts": {
            "conduit": int(np.sum(edge_types == 0)),
            "pump": int(np.sum(edge_types == 1)),
            "orifice": int(np.sum(edge_types == 2)),
            "weir": int(np.sum(edge_types == 3)),
        },
        "aggregated_graph_edge_count": int(graph.number_of_edges()),
        "community_count": int(new_graph.number_of_nodes()),
        "community_edge_count": int(new_graph.number_of_edges()),
        "community_positions_shape": list(community_pos.shape),
        "x_new_shape": list(x_new.shape),
        "edge_index_new_shape": list(edge_index_new.shape),
        "edge_weight_new_shape": list(edge_weight_new.shape),
        "cluster_assignments_shape": list(cluster_assignments.shape),
        "community_sizes": community_sizes,
    }
    (OUTPUT_DIR / "cluster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
