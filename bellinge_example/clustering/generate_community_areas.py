from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACK_ROOT = SCRIPT_DIR.parent
INP_PATH = PACK_ROOT / "swmm" / "bellinge_final.inp"
RAW_DIR = SCRIPT_DIR / "raw_graph"
CLUSTER_DIR = SCRIPT_DIR / "community_output"


def read_subcatchment_areas(inp_path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    areas: list[float] = []
    in_subcatchments = False
    for raw_line in inp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_subcatchments = line.upper() == "[SUBCATCHMENTS]"
            continue
        if not in_subcatchments or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        names.append(parts[0])
        areas.append(float(parts[3]))
    if not names:
        raise RuntimeError(f"Failed to read [SUBCATCHMENTS] from {inp_path}")
    return names, np.array(areas, dtype=float)


def main() -> None:
    cluster_assignments = np.load(CLUSTER_DIR / "cluster_assignments.npy").astype(int)
    node_lid_index = np.load(RAW_DIR / "NodeLidIndex.npy").astype(int).reshape(-1)
    saved_subcatchment_names = np.load(RAW_DIR / "subcatchment_names.npy", allow_pickle=True).astype(str)
    inp_subcatchment_names, subcatchment_areas = read_subcatchment_areas(INP_PATH)
    if list(saved_subcatchment_names) != inp_subcatchment_names:
        raise RuntimeError("subcatchment_names.npy order does not match INP [SUBCATCHMENTS].")
    if node_lid_index.shape[0] != subcatchment_areas.shape[0]:
        raise RuntimeError("NodeLidIndex.npy count does not match INP subcatchments.")
    node_to_community_0based = {
        int(node_index): int(community_id) for community_id, node_index in cluster_assignments
    }
    community_count = int(np.max(cluster_assignments[:, 0])) + 1
    subcatchment_category = np.zeros(subcatchment_areas.shape[0], dtype=int)
    community_total_area = np.zeros(community_count, dtype=float)
    for sub_index, outlet_node_index_1based in enumerate(node_lid_index):
        outlet_node_index_0based = int(outlet_node_index_1based) - 1
        community_id_0based = node_to_community_0based[outlet_node_index_0based]
        subcatchment_category[sub_index] = community_id_0based + 1
        community_total_area[community_id_0based] += subcatchment_areas[sub_index]
    np.save(CLUSTER_DIR / "subcatchment_community.npy", subcatchment_category)
    np.save(CLUSTER_DIR / "community_total_area.npy", community_total_area)
    summary = {
        "inp": str(INP_PATH),
        "output_dir": str(CLUSTER_DIR),
        "subcatchment_count": int(subcatchment_areas.shape[0]),
        "community_count": int(community_count),
        "category_index_base": "1-based community id",
        "subcatchment_category_shape": list(subcatchment_category.shape),
        "community_total_area_shape": list(community_total_area.shape),
        "total_subcatchment_area": float(np.sum(subcatchment_areas)),
        "total_community_area": float(np.sum(community_total_area)),
        "outputs": {
            "subcatchment_community": str(CLUSTER_DIR / "subcatchment_community.npy"),
            "community_total_area": str(CLUSTER_DIR / "community_total_area.npy"),
        },
    }
    (CLUSTER_DIR / "community_area_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
