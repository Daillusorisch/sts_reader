from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def run_once(
    path: Path, reader_module
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    t0 = time.perf_counter()
    times, data = reader_module.read_sts(str(path))
    dt = time.perf_counter() - t0
    time_shape = tuple(times.shape)
    data_shape = tuple(data.shape)
    del times
    del data
    gc.collect()
    return dt, time_shape, data_shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sts_reader.read_sts")
    parser.add_argument("file", type=Path, help="Path to .sts file")
    parser.add_argument("-n", "--repeat", type=int, default=7, help="Measured runs")
    parser.add_argument("-w", "--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument(
        "--no-validate-shape",
        action="store_true",
        help="Skip shape validation (default validates data second dim == 11)",
    )
    args = parser.parse_args()

    path = args.file
    if not path.exists():
        raise FileNotFoundError(path)

    import sts_reader as reader_module

    file_mb = path.stat().st_size / (1024 * 1024)
    print(f"File: {path}")
    print(f"Size: {file_mb:.2f} MiB")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")

    for i in range(args.warmup):
        dt, t_shape, d_shape = run_once(path, reader_module)
        print(f"[warmup {i + 1}] {dt:.4f}s, times={t_shape}, data={d_shape}")

    times_sec: list[float] = []
    last_time_shape: tuple[int, ...] | None = None
    last_data_shape: tuple[int, ...] | None = None

    for i in range(args.repeat):
        dt, t_shape, d_shape = run_once(path, reader_module)
        times_sec.append(dt)
        last_time_shape = t_shape
        last_data_shape = d_shape
        mbps = file_mb / dt if dt > 0 else float("inf")
        print(f"[run {i + 1}] {dt:.4f}s ({mbps:.2f} MiB/s)")

    sorted_t = sorted(times_sec)
    t_min = sorted_t[0]
    t_max = sorted_t[-1]
    t_med = statistics.median(sorted_t)
    t_avg = statistics.fmean(sorted_t)
    t_p95 = percentile(sorted_t, 0.95)

    print("\n=== Summary ===")
    print(f"shape(times): {last_time_shape}")
    print(f"shape(data):  {last_data_shape}")
    print(f"min:    {t_min:.4f}s ({file_mb / t_min:.2f} MiB/s)")
    print(f"median: {t_med:.4f}s ({file_mb / t_med:.2f} MiB/s)")
    print(f"avg:    {t_avg:.4f}s ({file_mb / t_avg:.2f} MiB/s)")
    print(f"p95:    {t_p95:.4f}s ({file_mb / t_p95:.2f} MiB/s)")
    print(f"max:    {t_max:.4f}s ({file_mb / t_max:.2f} MiB/s)")

    if not args.no_validate_shape:
        if (
            last_data_shape is None
            or len(last_data_shape) != 2
            or last_data_shape[1] != 11
        ):
            raise RuntimeError(
                f"Unexpected data shape: {last_data_shape}, expected (N, 11)"
            )
        if last_time_shape is None or len(last_time_shape) != 1:
            raise RuntimeError(
                f"Unexpected times shape: {last_time_shape}, expected (N,)"
            )
        if last_time_shape[0] != last_data_shape[0]:
            raise RuntimeError(
                f"Row count mismatch: times={last_time_shape[0]}, data={last_data_shape[0]}"
            )


if __name__ == "__main__":
    main()
