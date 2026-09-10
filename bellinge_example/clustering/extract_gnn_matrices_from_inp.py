from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from pyswmm import Simulation


SCRIPT_DIR = Path(__file__).resolve().parent
PACK_ROOT = SCRIPT_DIR.parent
DEFAULT_INP = PACK_ROOT / "swmm" / "bellinge_final.inp"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "raw_graph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DEFAULT_INP))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def validate_with_pyswmm(inp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pyswmm_validate_") as temp_dir:
        temp_inp = Path(temp_dir) / "model.inp"
        temp_rpt = Path(temp_dir) / "model.rpt"
        temp_out = Path(temp_dir) / "model.out"
        shutil.copy2(inp_path, temp_inp)
        with Simulation(str(temp_inp), reportfile=str(temp_rpt), outputfile=str(temp_out)):
            pass


def read_inp_sections(inp_path: Path) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current_section: str | None = None
    for raw_line in inp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").upper()
            sections.setdefault(current_section, [])
            continue
        if current_section is None or line.startswith(";"):
            continue
        parts = line.split()
        if parts:
            sections[current_section].append(parts)
    return sections


def to_float(value: str) -> float:
    return float(value)


def min_max_norm(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    value_min = np.min(values)
    value_max = np.max(values)
    if value_max == value_min:
        return np.zeros_like(values, dtype=float)
    return (values - value_min) / (value_max - value_min)


def build_node_table(sections: dict[str, list[list[str]]]) -> tuple[list[str], np.ndarray, dict[str, int]]:
    coordinates = sections.get("COORDINATES", [])
    if not coordinates:
        raise RuntimeError("Missing [COORDINATES].")
    node_names = [row[0] for row in coordinates]
    coords = np.array([[to_float(row[1]), to_float(row[2])] for row in coordinates], dtype=float)
    node_to_index = {name: index for index, name in enumerate(node_names)}
    return node_names, coords, node_to_index


def build_edges(
    sections: dict[str, list[list[str]]],
    node_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    conduits = sections.get("CONDUITS", [])
    if not conduits:
        raise RuntimeError("Missing [CONDUITS].")
    edge_rows: list[list[int]] = []
    edge_names: list[str] = []
    edge_types: list[int] = []
    missing_nodes: list[str] = []
    conduit_lengths = [to_float(row[3]) for row in conduits]
    conduit_weights = min_max_norm(np.array(conduit_lengths, dtype=float))
    non_conduit_weight = float(np.median(conduit_weights))
    edge_weights: list[float] = []

    def add_link(row: list[str], edge_type: int, weight: float) -> None:
        link_name, from_node, to_node = row[0], row[1], row[2]
        if from_node not in node_to_index or to_node not in node_to_index:
            missing_nodes.extend([node for node in (from_node, to_node) if node not in node_to_index])
            return
        edge_rows.append([node_to_index[from_node] + 1, node_to_index[to_node] + 1])
        edge_weights.append(weight)
        edge_types.append(edge_type)
        edge_names.append(link_name)

    for row, weight in zip(conduits, conduit_weights):
        add_link(row, edge_type=0, weight=float(weight))
    for row in sections.get("PUMPS", []):
        add_link(row, edge_type=1, weight=non_conduit_weight)
    for row in sections.get("ORIFICES", []):
        add_link(row, edge_type=2, weight=non_conduit_weight)
    for row in sections.get("WEIRS", []):
        add_link(row, edge_type=3, weight=non_conduit_weight)
    if missing_nodes:
        unique_missing = sorted(set(missing_nodes))
        raise RuntimeError(f"Link nodes missing from [COORDINATES]: {unique_missing[:20]}")
    return (
        np.array(edge_rows, dtype=float),
        np.array(edge_weights, dtype=float),
        np.array(edge_types, dtype=int),
        edge_names,
    )


def build_subcatchment_features(
    sections: dict[str, list[list[str]]],
    node_names: list[str],
    node_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    subcatchments = sections.get("SUBCATCHMENTS", [])
    x_base = np.zeros((len(node_names), 4), dtype=float)
    if not subcatchments:
        return x_base, np.array([], dtype=float), []
    sub_names = [row[0] for row in subcatchments]
    outlets = [row[2] for row in subcatchments]
    areas = np.array([to_float(row[3]) for row in subcatchments], dtype=float)
    impervious = np.array([to_float(row[4]) for row in subcatchments], dtype=float) / 100.0
    widths = np.array([to_float(row[5]) for row in subcatchments], dtype=float)
    slopes = np.array([to_float(row[6]) for row in subcatchments], dtype=float) / 100.0
    areas_norm = min_max_norm(areas)
    widths_norm = min_max_norm(widths)
    node_lid_index: list[int] = []
    for sub_index, outlet in enumerate(outlets):
        if outlet not in node_to_index:
            raise RuntimeError(f"Subcatchment {sub_names[sub_index]} outlet not in coordinates: {outlet}")
        node_idx = node_to_index[outlet]
        node_lid_index.append(node_idx + 1)
        x_base[node_idx, 0] += areas_norm[sub_index]
        x_base[node_idx, 2] += widths_norm[sub_index]
        x_base[node_idx, 1] = max(x_base[node_idx, 1], impervious[sub_index])
        x_base[node_idx, 3] = max(x_base[node_idx, 3], slopes[sub_index])
    return x_base, np.array(node_lid_index, dtype=float).reshape(-1, 1), sub_names


def collect_node_hydraulic_attributes(
    sections: dict[str, list[list[str]]],
) -> tuple[dict[str, float], dict[str, float], set[str]]:
    elevations: dict[str, float] = {}
    max_depths: dict[str, float] = {}
    outfalls: set[str] = set()
    for row in sections.get("JUNCTIONS", []):
        elevations[row[0]] = to_float(row[1])
        max_depths[row[0]] = to_float(row[2])
    for row in sections.get("STORAGE", []):
        elevations[row[0]] = to_float(row[1])
        max_depths[row[0]] = to_float(row[2])
    for row in sections.get("OUTFALLS", []):
        elevations[row[0]] = to_float(row[1])
        max_depths[row[0]] = 10.0
        outfalls.add(row[0])
    return elevations, max_depths, outfalls


def build_x_combine_final(
    sections: dict[str, list[list[str]]],
    node_names: list[str],
    coords: np.ndarray,
    node_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_base, node_lid_index, subcatchment_names = build_subcatchment_features(sections, node_names, node_to_index)
    elevations, max_depths, outfalls = collect_node_hydraulic_attributes(sections)
    missing_attributes = [name for name in node_names if name not in elevations or name not in max_depths]
    if missing_attributes:
        raise RuntimeError(f"Nodes missing invert elevation or max depth: {missing_attributes[:20]}")
    outfall_flag = np.array([1.0 if name in outfalls else 0.0 for name in node_names], dtype=float).reshape(-1, 1)
    coord_x_norm = min_max_norm(coords[:, 0]).reshape(-1, 1)
    coord_y_norm = min_max_norm(coords[:, 1]).reshape(-1, 1)
    elevation_norm = min_max_norm(np.array([elevations[name] for name in node_names], dtype=float)).reshape(-1, 1)
    max_depth_norm = min_max_norm(np.array([max_depths[name] for name in node_names], dtype=float)).reshape(-1, 1)
    x_combine_final = np.hstack(
        [x_base, outfall_flag, coord_x_norm, coord_y_norm, elevation_norm, max_depth_norm]
    )
    return x_combine_final, node_lid_index, subcatchment_names


def main() -> None:
    args = parse_args()
    inp_path = Path(args.inp).resolve()
    output_dir = Path(args.out_dir).resolve()
    if not inp_path.exists():
        raise FileNotFoundError(f"INP not found: {inp_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_with_pyswmm(inp_path)
    sections = read_inp_sections(inp_path)
    node_names, coords, node_to_index = build_node_table(sections)
    edges, edges_weight, edge_types, edge_names = build_edges(sections, node_to_index)
    x_combine_final, node_lid_index, subcatchment_names = build_x_combine_final(
        sections, node_names, coords, node_to_index
    )
    for stale_name in ("edges.npy", "edges_weight.npy", "conduit_names.npy"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    np.save(output_dir / "edges_all_links.npy", edges)
    np.save(output_dir / "edges_weight_all_links.npy", edges_weight)
    np.save(output_dir / "edge_type_all_links.npy", edge_types)
    np.save(output_dir / "X_combine_final.npy", x_combine_final)
    np.save(output_dir / "NodeLidIndex.npy", node_lid_index)
    np.save(output_dir / "node_names.npy", np.array(node_names, dtype=str))
    np.save(output_dir / "edge_names_all_links.npy", np.array(edge_names, dtype=str))
    np.save(output_dir / "subcatchment_names.npy", np.array(subcatchment_names, dtype=str))
    summary = {
        "inp": str(inp_path),
        "output_dir": str(output_dir),
        "node_count": len(node_names),
        "all_link_edge_count": int(edges.shape[0]),
        "edge_type_counts": {
            "conduit": int(np.sum(edge_types == 0)),
            "pump": int(np.sum(edge_types == 1)),
            "orifice": int(np.sum(edge_types == 2)),
            "weir": int(np.sum(edge_types == 3)),
        },
        "subcatchment_count": len(subcatchment_names),
        "edges_all_links_shape": list(edges.shape),
        "edges_weight_all_links_shape": list(edges_weight.shape),
        "edge_type_all_links_shape": list(edge_types.shape),
        "X_combine_final_shape": list(x_combine_final.shape),
        "NodeLidIndex_shape": list(node_lid_index.shape),
        "edge_index_base": "1-based",
        "edge_source": "CONDUITS + PUMPS + ORIFICES + WEIRS",
        "edge_type_codes": {"0": "conduit", "1": "pump", "2": "orifice", "3": "weir"},
        "edge_weight_rule": "conduits use min-max normalized length; pumps/orifices/weirs use median conduit weight",
        "x_columns": [
            "normalized subcatchment area sum",
            "max imperviousness / 100",
            "normalized subcatchment width sum",
            "max slope / 100",
            "is outfall",
            "normalized node x",
            "normalized node y",
            "normalized node invert elevation",
            "normalized node max depth",
        ],
    }
    (output_dir / "matrix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
