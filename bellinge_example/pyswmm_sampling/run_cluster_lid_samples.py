from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from cluster_lid_pyswmm import (
    PACK_ROOT,
    load_cluster_arrays,
    load_model_metadata,
    run_one_sample,
    save_batch_outputs,
)

DEFAULT_SAMPLES = PACK_ROOT / "lid_sampling" / "samples1_3d.npy"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_sample_task(task: tuple[int, np.ndarray, str, bool]) -> tuple:
    sample_id, sample, output_dir, keep_rpt = task
    metadata = load_model_metadata()
    subcatchment_categories, community_total_area = load_cluster_arrays()
    return run_one_sample(
        sample_id=sample_id,
        sample=sample,
        metadata=metadata,
        subcatchment_categories_1based=subcatchment_categories,
        community_total_area=community_total_area,
        output_dir=Path(output_dir),
        keep_rpt=keep_rpt,
    )


def iter_batches(start: int, end: int, batch_size: int) -> list[tuple[int, int]]:
    return [(batch_start, min(batch_start + batch_size, end)) for batch_start in range(start, end, batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--keep-rpt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples).resolve()
    output_dir = Path(args.output_dir).resolve()
    samples = np.load(samples_path)
    if samples.ndim != 3 or samples.shape[1:] != (47, 3):
        raise RuntimeError(f"samples1_3d.npy shape should be [N, 47, 3], got {samples.shape}")
    start = args.start
    end = min(start + args.count, samples.shape[0])
    if start < 0 or start >= end:
        raise RuntimeError(f"invalid sample range: start={start}, count={args.count}, total={samples.shape[0]}")
    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    output_names = None
    batch_start = time.perf_counter()
    progress = create_progress(total=end - start)
    for current_batch_start, current_batch_end in iter_batches(start, end, args.batch_size):
        batch_dir = output_dir / "batches" / f"batch_{current_batch_start:05d}_{current_batch_end - 1:05d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            (sample_id, samples[sample_id], str(batch_dir), args.keep_rpt)
            for sample_id in range(current_batch_start, current_batch_end)
        ]
        results = run_batch(tasks, args.workers, progress)
        results.sort(key=lambda item: item[0].sample_id)
        all_results.extend(result for result, _ in results)
        batch_names = next((names for _, names in results if names is not None), None)
        if batch_names is None:
            save_failure_summary(
                [result for result, _ in results],
                batch_dir,
                samples_path,
                current_batch_start,
                current_batch_end,
                0.0,
            )
        else:
            output_names = batch_names
            save_batch_outputs([result for result, _ in results], batch_names, batch_dir)
        write_progress_message(
            progress,
            json.dumps(
                {
                    "batch_start": current_batch_start,
                    "batch_end_exclusive": current_batch_end,
                    "success_count": sum(1 for result, _ in results if result.success),
                    "batch_dir": str(batch_dir),
                }
            ),
        )
    close_progress(progress)
    if output_names is None:
        save_failure_summary(all_results, output_dir, samples_path, start, end, time.perf_counter() - batch_start)
        raise RuntimeError("All samples failed.")
    write_run_metadata(
        output_dir,
        samples_path,
        samples.shape,
        start,
        end,
        time.perf_counter() - batch_start,
        args.workers,
        args.batch_size,
        args.keep_rpt,
        all_results,
    )


def create_progress(total: int):
    if tqdm is None:
        return None
    return tqdm(total=total, desc="Running SWMM samples", unit="sample", dynamic_ncols=True)


def close_progress(progress) -> None:
    if progress is not None:
        progress.close()


def write_progress_message(progress, message: str) -> None:
    if progress is None:
        print(message)
    else:
        progress.write(message)


def update_progress(progress, result) -> None:
    if progress is None:
        print_sample_status(result)
        return
    progress.update(1)
    progress.set_postfix(
        {
            "last": result.sample_id,
            "ok": int(result.success),
            "sec": f"{result.elapsed_seconds:.1f}",
        }
    )
    if not result.success:
        write_progress_message(
            progress,
            json.dumps(
                {
                    "sample_id": result.sample_id,
                    "success": result.success,
                    "elapsed_seconds": result.elapsed_seconds,
                    "message": result.message,
                }
            ),
        )


def run_batch(tasks: list[tuple[int, np.ndarray, str, bool]], workers: int, progress=None) -> list[tuple]:
    if workers == 1:
        results = []
        for task in tasks:
            result, names = run_sample_task(task)
            update_progress(progress, result)
            results.append((result, names))
        return results
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_sample = {executor.submit(run_sample_task, task): int(task[0]) for task in tasks}
        for future in as_completed(future_to_sample):
            result, names = future.result()
            update_progress(progress, result)
            results.append((result, names))
    return results


def print_sample_status(result) -> None:
    print(
        json.dumps(
            {
                "sample_id": result.sample_id,
                "success": result.success,
                "elapsed_seconds": result.elapsed_seconds,
                "message": result.message,
            }
        )
    )


def save_failure_summary(
    results: list,
    output_dir: Path,
    samples_path: Path,
    start: int,
    end: int,
    elapsed_seconds: float,
) -> None:
    summary = {
        "samples_path": str(samples_path),
        "start": start,
        "end_exclusive": end,
        "elapsed_seconds": elapsed_seconds,
        "success_count": 0,
        "failures": [
            {
                "sample_id": result.sample_id,
                "elapsed_seconds": result.elapsed_seconds,
                "message": result.message,
            }
            for result in results
        ],
    }
    (output_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_run_metadata(
    output_dir: Path,
    samples_path: Path,
    samples_shape: tuple[int, ...],
    start: int,
    end: int,
    elapsed_seconds: float,
    workers: int,
    batch_size: int,
    keep_rpt: bool,
    results: list,
) -> None:
    success_count = sum(1 for result in results if result.success)
    metadata = {
        "samples_path": str(samples_path),
        "samples_shape": list(samples_shape),
        "start": start,
        "end_exclusive": end,
        "run_count": end - start,
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "workers": workers,
        "batch_size": batch_size,
        "keep_rpt": keep_rpt,
        "elapsed_seconds": elapsed_seconds,
        "sample_unit": "0-1 ratio of each community area",
        "lid_types_order": ["RG", "brc", "pp"],
        "batch_output_dir": str(output_dir / "batches"),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output_dir / "run_status_all.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "success", "elapsed_seconds", "message"])
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.sample_id):
            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "success": int(result.success),
                    "elapsed_seconds": f"{result.elapsed_seconds:.6f}",
                    "message": result.message,
                }
            )


if __name__ == "__main__":
    main()
