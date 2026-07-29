"""ctypes loader for the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_LIBE57_LIB") or os.path.join(
    ROOT, "dist", "libmojo-libe57.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "e57_encoded_size": ([I, I, I], I),
    "e57_encode_i64": ([I, I, I, I, I, I], I),
    "e57_decode_i64": ([I, I, I, I, I, I], I),
    "e57_quantize_f64": ([I, I, I, I, F, F, I], I),
    "e57_dequantize_f64": ([I, I, F, F, I], I),
    "e57_encode_f32": ([I, I, I, I], I),
    "e57_decode_f32": ([I, I, I, I], I),
    "e57_encode_f64": ([I, I, I, I], I),
    "e57_decode_f64": ([I, I, I, I], I),
    "e57_encode_strings": ([I, I, I, I, I, I], I),
    "e57_decode_strings": ([I, I, I, I, I, I], I),
    "e57_spherical_to_cartesian": ([I, I, I], I),
    "e57_cartesian_to_spherical": ([I, I, I], I),
    "e57_apply_pose": ([I, I, I] + [F] * 7, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_LIBE57_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(directory, name)
        for directory, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run through `pixi run`")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return array.ctypes.data
