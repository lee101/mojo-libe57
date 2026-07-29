"""Benchmarks against source-derived NumPy and the pye57 pose reference."""

from __future__ import annotations

import os
import platform
import statistics
import time

import numpy as np
import pye57

import mojo_libe57 as e57


def median_seconds(function, repeats: int = 7) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def numpy_decode_12(encoded: bytes, count: int) -> np.ndarray:
    bits = np.unpackbits(
        np.frombuffer(encoded, dtype=np.uint8), bitorder="little", count=count * 12
    ).reshape(count, 12)
    return bits @ (np.uint64(1) << np.arange(12, dtype=np.uint64))


def numpy_spherical(points: np.ndarray) -> np.ndarray:
    result = np.empty_like(points)
    horizontal = points[:, 0] * np.cos(points[:, 2])
    result[:, 0] = horizontal * np.cos(points[:, 1])
    result[:, 1] = horizontal * np.sin(points[:, 1])
    result[:, 2] = points[:, 0] * np.sin(points[:, 2])
    return result


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(57)
    count = 1_000_000

    integers = rng.integers(0, 4096, size=count, dtype=np.int64)
    encoded = e57.encode_integers(integers, 0, 4095)
    np.testing.assert_array_equal(
        e57.decode_integers(encoded, count, 0, 4095),
        numpy_decode_12(encoded, count),
    )

    spherical = np.empty((count, 3), dtype=np.float64)
    spherical[:, 0] = rng.uniform(0.0, 200.0, count)
    spherical[:, 1] = rng.uniform(-np.pi, np.pi, count)
    spherical[:, 2] = rng.uniform(-np.pi / 2, np.pi / 2, count)
    np.testing.assert_allclose(
        e57.spherical_to_cartesian(spherical),
        numpy_spherical(spherical),
        rtol=3e-15,
        atol=3e-14,
    )
    points = rng.normal(size=(count, 3))
    angle = 0.73
    rotation = np.array(
        [
            np.cos(angle / 2),
            np.sin(angle / 2) / np.sqrt(3),
            np.sin(angle / 2) / np.sqrt(3),
            np.sin(angle / 2) / np.sqrt(3),
        ]
    )
    translation = np.array([10.0, -2.0, 0.25])
    pose = e57.ScanPose(tuple(rotation), tuple(translation))
    np.testing.assert_allclose(
        e57.apply_pose(points, pose),
        pye57.E57.to_global(points, rotation, translation),
        rtol=2e-15,
        atol=2e-15,
    )

    cases = [
        (
            "12-bit integer decode",
            lambda: e57.decode_integers(encoded, count, 0, 4095),
            lambda: numpy_decode_12(encoded, count),
            "source-derived NumPy",
        ),
        (
            "spherical to Cartesian",
            lambda: e57.spherical_to_cartesian(spherical),
            lambda: numpy_spherical(spherical),
            "NumPy",
        ),
        (
            "scan pose",
            lambda: e57.apply_pose(points, pose),
            lambda: pye57.E57.to_global(points, rotation, translation),
            "pye57/NumPy",
        ),
    ]
    print(f"Machine: {cpu_name()}, {os.cpu_count()} logical CPUs, {platform.platform()}")
    print(f"Dataset: {count:,} points/records; median of 7 runs")
    print()
    print("| Kernel | Mojo ms | Reference | Reference ms | Speedup |")
    print("|---|---:|---|---:|---:|")
    for name, mojo_function, reference_function, reference_name in cases:
        mojo_time = median_seconds(mojo_function)
        reference_time = median_seconds(reference_function)
        print(
            f"| {name} | {mojo_time * 1000:.3f} | {reference_name} | "
            f"{reference_time * 1000:.3f} | {reference_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
