from __future__ import annotations

import csv
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyswmm import Simulation
from swmm.toolkit import output
from swmm.toolkit.shared_enum import ElementType, LinkResult, NodeResult, SubcatchResult, Time

PACK_ROOT = Path(__file__).resolve().parent.parent
FINAL_INP = PACK_ROOT / "swmm" / "bellinge_final.inp"
CLUSTER_DIR = PACK_ROOT / "clustering" / "community_output"
LID_TYPES = ("RG", "brc", "pp")
AREA_TO_SQUARE_METERS = 10000.0


@dataclass(frozen=True)
class SubcatchmentInfo:
    names: list[str]
    outlet_nodes: list[str]
    areas: np.ndarray


@dataclass(frozen=True)
class ModelMetadata:
    subcatchments: SubcatchmentInfo
    lid_usage_start: int
    lid_usage_next_section: int
    lid_usage_header_end: int
    lines: list[str]


@dataclass
class SampleResult:
    sample_id: int
    success: bool
    elapsed_seconds: float
    message: str
    data_yl: np.ndarray | None = None
    subcatchment_runoff: np.ndarray | None = None
    node_inflow: np.ndarray | None = None
    node_flooding_series: np.ndarray | None = None
    node_depth: np.ndarray | None = None
    link_flow: np.ndarray | None = None


def read_inp_lines(inp_path: Path) -> list[str]:
    return inp_path.read_text(encoding="utf-8", errors="ignore").splitlines()


def find_section(lines: list[str], section_name: str) -> tuple[int, int]:
    target = f"[{section_name.upper()}]"
    start = next((i for i, line in enumerate(lines) if line.strip().upper() == target), None)
    if start is None:
        raise RuntimeError(f"Missing INP section {target}")
    next_section = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[") and lines[i].strip().endswith("]")),
        len(lines),
    )
    return start, next_section


def parse_subcatchments(lines: list[str]) -> SubcatchmentInfo:
    start, end = find_section(lines, "SUBCATCHMENTS")
    names: list[str] = []
    outlet_nodes: list[str] = []
    areas: list[float] = []
    for line in lines[start + 1 : end]:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        names.append(parts[0])
        outlet_nodes.append(parts[2])
        areas.append(float(parts[3]))
    if not names:
        raise RuntimeError("[SUBCATCHMENTS] is empty.")
    return SubcatchmentInfo(names=names, outlet_nodes=outlet_nodes, areas=np.array(areas, dtype=float))


def load_model_metadata(inp_path: Path = FINAL_INP) -> ModelMetadata:
    lines = read_inp_lines(inp_path)
    lid_start, lid_next_section = find_section(lines, "LID_USAGE")
    header_end = lid_start + 1
    while header_end < lid_next_section:
        stripped = lines[header_end].strip()
        if stripped and not stripped.startswith(";"):
            break
        header_end += 1
    return ModelMetadata(
        subcatchments=parse_subcatchments(lines),
        lid_usage_start=lid_start,
        lid_usage_next_section=lid_next_section,
        lid_usage_header_end=header_end,
        lines=lines,
    )


def load_cluster_arrays() -> tuple[np.ndarray, np.ndarray]:
    categories = np.load(CLUSTER_DIR / "subcatchment_community.npy").astype(int).reshape(-1)
    community_area = np.load(CLUSTER_DIR / "community_total_area.npy").astype(float).reshape(-1)
    return categories, community_area


def allocate_lid_area_to_subcatchments(
    sample: np.ndarray,
    subcatchment_areas: np.ndarray,
    subcatchment_categories_1based: np.ndarray,
    community_total_area: np.ndarray,
    *,
    sample_is_ratio: bool = True,
) -> np.ndarray:
    sample = np.asarray(sample, dtype=float)
    if sample.shape != (community_total_area.shape[0], len(LID_TYPES)):
        raise ValueError(f"sample shape should be {(community_total_area.shape[0], len(LID_TYPES))}, got {sample.shape}")
    if sample_is_ratio:
        community_lid_area = sample * community_total_area[:, None]
    else:
        community_lid_area = sample.copy()
    lid_area = np.zeros((subcatchment_areas.shape[0], len(LID_TYPES)), dtype=float)
    for sub_idx, community_id_1based in enumerate(subcatchment_categories_1based):
        community_idx = int(community_id_1based) - 1
        total_area = community_total_area[community_idx]
        if total_area <= 0:
            continue
        area_ratio = subcatchment_areas[sub_idx] / total_area
        lid_area[sub_idx, :] = community_lid_area[community_idx, :] * area_ratio
    total_lid_area = np.sum(lid_area, axis=1)
    invalid = np.where(total_lid_area > subcatchment_areas + 1e-8)[0]
    if invalid.size:
        first = int(invalid[0])
        raise ValueError(
            f"subcatchment {first + 1} total LID area {total_lid_area[first]:.6g} "
            f"exceeds subcatchment area {subcatchment_areas[first]:.6g}"
        )
    return lid_area


def build_lid_usage_lines(subcatchment_names: list[str], lid_area: np.ndarray) -> list[str]:
    usage_lines: list[str] = []
    for sub_idx, subcatchment_name in enumerate(subcatchment_names):
        for lid_idx, lid_name in enumerate(LID_TYPES):
            area_m2 = lid_area[sub_idx, lid_idx] * AREA_TO_SQUARE_METERS
            usage_lines.append(
                f"{subcatchment_name:<16} {lid_name:<16} 1       {area_m2:.10g} 0          0          0          0"
            )
    return usage_lines


def render_inp_with_lid_usage(metadata: ModelMetadata, lid_area: np.ndarray) -> str:
    usage_lines = build_lid_usage_lines(metadata.subcatchments.names, lid_area)
    new_lines = (
        metadata.lines[: metadata.lid_usage_header_end]
        + usage_lines
        + [""]
        + metadata.lines[metadata.lid_usage_next_section :]
    )
    return "\n".join(new_lines) + "\n"


def run_swmm(inp_text: str, work_dir: Path) -> tuple[Path, Path]:
    inp_path = work_dir / "model.inp"
    rpt_path = work_dir / "model.rpt"
    out_path = work_dir / "model.out"
    inp_path.write_text(inp_text, encoding="utf-8")
    with Simulation(str(inp_path), reportfile=str(rpt_path), outputfile=str(out_path)) as sim:
        for _ in sim:
            pass
    return rpt_path, out_path


def _series_to_array(series: object) -> np.ndarray:
    if isinstance(series, tuple):
        series = series[0]
    return np.asarray(series, dtype=float)


def _element_names(handle: object, element_type: ElementType, count: int) -> list[str]:
    return [str(output.get_elem_name(handle, element_type, idx)) for idx in range(count)]


def extract_output_series(out_path: Path) -> tuple[dict[str, np.ndarray], dict[str, list[str] | int]]:
    handle = output.init()
    try:
        output.open(handle, str(out_path))
        sizes = output.get_proj_size(handle)
        subcatch_count, node_count, link_count = int(sizes[0]), int(sizes[1]), int(sizes[2])
        period_count = int(output.get_times(handle, Time.NUM_PERIODS))
        start_period = 0
        end_period = period_count - 1
        subcatchment_runoff = np.vstack(
            [
                _series_to_array(output.get_subcatch_series(handle, idx, SubcatchResult.RUNOFF, start_period, end_period))
                for idx in range(subcatch_count)
            ]
        )
        node_flooding = np.vstack(
            [
                _series_to_array(output.get_node_series(handle, idx, NodeResult.FLOOD, start_period, end_period))
                for idx in range(node_count)
            ]
        )
        node_inflow = np.vstack(
            [
                _series_to_array(output.get_node_series(handle, idx, NodeResult.TOTAL_INFLOW, start_period, end_period))
                for idx in range(node_count)
            ]
        )
        node_depth = np.vstack(
            [
                _series_to_array(output.get_node_series(handle, idx, NodeResult.DEPTH, start_period, end_period))
                for idx in range(node_count)
            ]
        )
        link_flow = np.vstack(
            [
                _series_to_array(output.get_link_series(handle, idx, LinkResult.FLOW, start_period, end_period))
                for idx in range(link_count)
            ]
        )
        data_yl = np.sum(node_flooding, axis=1)
        names = {
            "subcatchment_names": _element_names(handle, ElementType.SUBCATCH, subcatch_count),
            "node_names": _element_names(handle, ElementType.NODE, node_count),
            "link_names": _element_names(handle, ElementType.LINK, link_count),
            "period_count": period_count,
        }
        data = {
            "data_yl": data_yl,
            "subcatchment_runoff": subcatchment_runoff,
            "node_inflow": node_inflow,
            "node_flooding_series": node_flooding,
            "node_depth": node_depth,
            "link_flow": link_flow,
        }
        return data, names
    finally:
        output.close(handle)


def run_one_sample(
    sample_id: int,
    sample: np.ndarray,
    metadata: ModelMetadata,
    subcatchment_categories_1based: np.ndarray,
    community_total_area: np.ndarray,
    output_dir: Path,
    *,
    keep_rpt: bool = True,
) -> tuple[SampleResult, dict[str, list[str] | int] | None]:
    start = time.perf_counter()
    sample_dir = output_dir / "sample_runs" / f"sample_{sample_id:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    try:
        lid_area = allocate_lid_area_to_subcatchments(
            sample,
            metadata.subcatchments.areas,
            subcatchment_categories_1based,
            community_total_area,
            sample_is_ratio=True,
        )
        inp_text = render_inp_with_lid_usage(metadata, lid_area)
        with tempfile.TemporaryDirectory(prefix=f"cluster_lid_{sample_id:05d}_") as temp_dir:
            _, temp_out = run_swmm(inp_text, Path(temp_dir))
            data, names = extract_output_series(temp_out)
            if keep_rpt:
                shutil.copy2(Path(temp_dir) / "model.rpt", sample_dir / "model.rpt")
        np.save(sample_dir / "lid_area_by_subcatchment.npy", lid_area)
        elapsed = time.perf_counter() - start
        return (
            SampleResult(
                sample_id=sample_id,
                success=True,
                elapsed_seconds=elapsed,
                message="ok",
                data_yl=data["data_yl"],
                subcatchment_runoff=data["subcatchment_runoff"],
                node_inflow=data["node_inflow"],
                node_flooding_series=data["node_flooding_series"],
                node_depth=data["node_depth"],
                link_flow=data["link_flow"],
            ),
            names,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return (
            SampleResult(
                sample_id=sample_id,
                success=False,
                elapsed_seconds=elapsed,
                message=repr(exc),
            ),
            None,
        )


def save_batch_outputs(results: list[SampleResult], names: dict[str, list[str] | int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = [result for result in results if result.success]
    if not successful:
        raise RuntimeError("No successful samples, cannot save result matrices.")

    def stack_or_nan(attr: str) -> np.ndarray:
        template = getattr(successful[0], attr)
        assert template is not None
        stacked = np.full((len(results), *template.shape), np.nan, dtype=float)
        for row_idx, result in enumerate(results):
            value = getattr(result, attr)
            if result.success and value is not None:
                stacked[row_idx] = value
        return stacked

    np.save(output_dir / "data_YL.npy", stack_or_nan("data_yl"))
    np.save(output_dir / "data_Subcatchment_outflow.npy", stack_or_nan("subcatchment_runoff"))
    np.save(output_dir / "data_Node_inflow.npy", stack_or_nan("node_inflow"))
    np.save(output_dir / "data_Node_flooding_series.npy", stack_or_nan("node_flooding_series"))
    np.save(output_dir / "data_Node_depth.npy", stack_or_nan("node_depth"))
    np.save(output_dir / "data_link_outflow.npy", stack_or_nan("link_flow"))
    np.save(output_dir / "subcatchment_names.npy", np.array(names["subcatchment_names"], dtype=str))
    np.save(output_dir / "node_names.npy", np.array(names["node_names"], dtype=str))
    np.save(output_dir / "link_names.npy", np.array(names["link_names"], dtype=str))
    with (output_dir / "run_status.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "success", "elapsed_seconds", "message"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "success": int(result.success),
                    "elapsed_seconds": f"{result.elapsed_seconds:.6f}",
                    "message": result.message,
                }
            )
    summary = {
        "sample_count": len(results),
        "success_count": len(successful),
        "period_count": int(names["period_count"]),
        "subcatchment_count": len(names["subcatchment_names"]),
        "node_count": len(names["node_names"]),
        "link_count": len(names["link_names"]),
        "outputs": {
            "data_YL": list(stack_or_nan("data_yl").shape),
            "data_Subcatchment_outflow": list(stack_or_nan("subcatchment_runoff").shape),
            "data_Node_inflow": list(stack_or_nan("node_inflow").shape),
            "data_Node_flooding_series": list(stack_or_nan("node_flooding_series").shape),
            "data_Node_depth": list(stack_or_nan("node_depth").shape),
            "data_link_outflow": list(stack_or_nan("link_flow").shape),
        },
    }
    (output_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
