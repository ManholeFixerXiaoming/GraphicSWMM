from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACK_ROOT = SCRIPT_DIR.parent
RAW_GRAPH_DIR = PACK_ROOT / "clustering" / "raw_graph"
CLUSTER_DIR = PACK_ROOT / "clustering" / "community_output"
FINAL_INP = PACK_ROOT / "swmm" / "bellinge_final.inp"
SAMPLES_PATH = PACK_ROOT / "lid_sampling" / "samples1_3d.npy"
OUTPUT_PATH = SCRIPT_DIR / "node_lid_inputs.npy"


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return path


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
        if len(parts) >= 4:
            names.append(parts[0])
            areas.append(float(parts[3]))
    if not names:
        raise RuntimeError(f"failed to read [SUBCATCHMENTS] from {inp_path}")
    return names, np.array(areas, dtype=np.float32)


def main() -> None:
    samples = np.load(require_file(SAMPLES_PATH)).astype(np.float32)
    subcatchment_categories = np.load(require_file(CLUSTER_DIR / "subcatchment_community.npy")).astype(int).reshape(-1)
    node_lid_index = np.load(require_file(RAW_GRAPH_DIR / "NodeLidIndex.npy")).astype(int).reshape(-1)
    subcatchment_names = np.load(require_file(RAW_GRAPH_DIR / "subcatchment_names.npy"), allow_pickle=True)
    node_names = np.load(require_file(RAW_GRAPH_DIR / "node_names.npy"), allow_pickle=True)
    inp_subcatchment_names, subcatchment_areas = read_subcatchment_areas(require_file(FINAL_INP))
    x_raw = np.load(require_file(RAW_GRAPH_DIR / "X_combine_final.npy")).astype(np.float32)
    if samples.ndim != 3 or samples.shape[1:] != (47, 3):
        raise RuntimeError(f"samples1_3d.npy should be [N, 47, 3], got {samples.shape}")
    if list(subcatchment_names.astype(str)) != inp_subcatchment_names:
        raise RuntimeError("subcatchment_names.npy order does not match INP [SUBCATCHMENTS].")
    node_count = node_names.shape[0]
    node_lid = np.zeros((samples.shape[0], node_count, 3), dtype=np.float32)
    for sub_idx, community_id_1based in enumerate(subcatchment_categories):
        community_idx = int(community_id_1based) - 1
        node_idx = int(node_lid_index[sub_idx]) - 1
        node_lid[:, node_idx, :] += samples[:, community_idx, :] * subcatchment_areas[sub_idx]
    np.save(OUTPUT_PATH, node_lid)
    summary = {
        "samples_shape": list(samples.shape),
        "node_lid_inputs_shape": list(node_lid.shape),
        "node_count": int(node_count),
        "subcatchment_count": int(subcatchment_names.shape[0]),
        "output": str(OUTPUT_PATH),
        "x_raw_shape": list(x_raw.shape),
        "total_subcatchment_area": float(subcatchment_areas.sum()),
    }
    (SCRIPT_DIR / "node_lid_inputs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
