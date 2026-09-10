from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw_graph"
CLUSTER_DIR = SCRIPT_DIR / "community_output"


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cluster_assignments = np.load(CLUSTER_DIR / "cluster_assignments.npy")
    node_names = np.load(RAW_DIR / "node_names.npy", allow_pickle=True)
    node_lid_index = np.load(RAW_DIR / "NodeLidIndex.npy").astype(int).reshape(-1)
    subcatchment_names = np.load(RAW_DIR / "subcatchment_names.npy", allow_pickle=True)
    if cluster_assignments.shape[1] != 2:
        raise RuntimeError("cluster_assignments.npy must have columns [community_id, node_index_0based].")
    if node_lid_index.shape[0] != subcatchment_names.shape[0]:
        raise RuntimeError("NodeLidIndex.npy and subcatchment_names.npy length mismatch.")
    return cluster_assignments, node_names, node_lid_index, subcatchment_names


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cluster_assignments, node_names, node_lid_index, subcatchment_names = load_inputs()
    node_to_community = {int(node_index): int(community_id) for community_id, node_index in cluster_assignments}
    community_to_nodes: dict[int, list[dict[str, object]]] = defaultdict(list)
    node_rows: list[dict[str, object]] = []
    for community_id, node_index in sorted(
        ((int(row[0]), int(row[1])) for row in cluster_assignments),
        key=lambda item: (item[0], item[1]),
    ):
        row = {
            "community_id": community_id,
            "node_index_0based": node_index,
            "node_index_1based": node_index + 1,
            "node_name": str(node_names[node_index]),
        }
        node_rows.append(row)
        community_to_nodes[community_id].append(row)
    subcatchment_rows: list[dict[str, object]] = []
    community_to_subcatchments: dict[int, list[dict[str, object]]] = defaultdict(list)
    for sub_index, outlet_node_index_1based in enumerate(node_lid_index):
        outlet_node_index_0based = int(outlet_node_index_1based) - 1
        community_id = node_to_community[outlet_node_index_0based]
        row = {
            "community_id": community_id,
            "subcatchment_index_0based": sub_index,
            "subcatchment_index_1based": sub_index + 1,
            "subcatchment_name": str(subcatchment_names[sub_index]),
            "outlet_node_index_0based": outlet_node_index_0based,
            "outlet_node_index_1based": int(outlet_node_index_1based),
            "outlet_node_name": str(node_names[outlet_node_index_0based]),
        }
        subcatchment_rows.append(row)
        community_to_subcatchments[community_id].append(row)
    subcatchment_rows.sort(key=lambda row: (int(row["community_id"]), int(row["subcatchment_index_0based"])))
    write_csv(
        CLUSTER_DIR / "community_nodes.csv",
        ["community_id", "node_index_0based", "node_index_1based", "node_name"],
        node_rows,
    )
    write_csv(
        CLUSTER_DIR / "community_subcatchments.csv",
        [
            "community_id",
            "subcatchment_index_0based",
            "subcatchment_index_1based",
            "subcatchment_name",
            "outlet_node_index_0based",
            "outlet_node_index_1based",
            "outlet_node_name",
        ],
        subcatchment_rows,
    )
    summary: dict[str, object] = {
        "cluster_output_dir": str(CLUSTER_DIR),
        "raw_matrix_dir": str(RAW_DIR),
        "reproducibility": {
            "python_random_seed": 1314,
            "numpy_random_seed": 1314,
            "louvain_random_state": 1314,
            "networkx_louvain_seed_fallback": 1314,
        },
        "community_count": len(community_to_nodes),
        "node_count": len(node_names),
        "subcatchment_count": len(subcatchment_names),
        "communities": {},
    }
    communities = summary["communities"]
    assert isinstance(communities, dict)
    for community_id in sorted(community_to_nodes):
        nodes = community_to_nodes[community_id]
        subcatchments = community_to_subcatchments.get(community_id, [])
        communities[str(community_id)] = {
            "node_count": len(nodes),
            "subcatchment_count": len(subcatchments),
            "nodes": [
                {
                    "node_index_0based": row["node_index_0based"],
                    "node_index_1based": row["node_index_1based"],
                    "node_name": row["node_name"],
                }
                for row in nodes
            ],
            "subcatchments": [
                {
                    "subcatchment_index_0based": row["subcatchment_index_0based"],
                    "subcatchment_index_1based": row["subcatchment_index_1based"],
                    "subcatchment_name": row["subcatchment_name"],
                    "outlet_node_index_1based": row["outlet_node_index_1based"],
                    "outlet_node_name": row["outlet_node_name"],
                }
                for row in subcatchments
            ],
        }
    (CLUSTER_DIR / "community_mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "community_nodes_csv": str(CLUSTER_DIR / "community_nodes.csv"),
                "community_subcatchments_csv": str(CLUSTER_DIR / "community_subcatchments.csv"),
                "community_mapping_summary_json": str(CLUSTER_DIR / "community_mapping_summary.json"),
                "community_count": len(community_to_nodes),
                "node_count": len(node_names),
                "subcatchment_count": len(subcatchment_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
