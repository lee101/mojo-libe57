"""Python API for E57 compressed-vector and point kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from ._lib import addr, lib


class E57CodecError(ValueError):
    pass


@dataclass(frozen=True)
class ScaledInteger:
    minimum: int
    maximum: int
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum < -(1 << 63) or self.maximum >= 1 << 63:
            raise ValueError("minimum and maximum must fit in int64")
        if self.maximum < self.minimum:
            raise ValueError("maximum must be at least minimum")
        if not np.isfinite(self.scale) or not np.isfinite(self.offset):
            raise ValueError("scale and offset must be finite")
        if self.scale == 0:
            raise ValueError("scale must be nonzero")


@dataclass(frozen=True)
class ScanPose:
    """E57 pose as quaternion `(w, x, y, z)` and translation in metres."""

    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.rotation) != 4 or len(self.translation) != 3:
            raise ValueError("rotation needs 4 values and translation needs 3")


_ERRORS = {
    -1: "invalid count, bounds, offsets, or scale",
    -2: "truncated input or destination buffer too small",
    -3: "value outside the declared E57 bounds",
    -4: "scaled value is not representable as int64",
}


def _check(code: int) -> int:
    if code < 0:
        raise E57CodecError(_ERRORS.get(code, f"codec error {code}"))
    return code


def _i64(values: Iterable[int] | np.ndarray) -> np.ndarray:
    source = np.asarray(values if isinstance(values, np.ndarray) else list(values))
    if source.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if source.size == 0:
        return np.empty(0, dtype=np.int64)
    if source.dtype.kind == "O":
        items = source.tolist()
        if not all(isinstance(value, (int, np.integer)) for value in items):
            raise TypeError("integer values are required")
        if any(value < -(1 << 63) or value > (1 << 63) - 1 for value in items):
            raise OverflowError("integer value does not fit in int64")
    elif source.dtype.kind not in "iu":
        raise TypeError("integer values are required")
    if source.size and (
        np.any(source < -(1 << 63)) or np.any(source > (1 << 63) - 1)
    ):
        raise OverflowError("integer value does not fit in int64")
    return np.ascontiguousarray(source, dtype=np.int64)


def _float_array(
    values: Iterable[float] | np.ndarray, dtype: np.dtype | type
) -> np.ndarray:
    source = np.asarray(values if isinstance(values, np.ndarray) else list(values))
    if source.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if source.dtype.kind not in "iuf":
        raise TypeError("numeric values are required")
    return np.ascontiguousarray(source, dtype=dtype)


def _bounds(minimum: int, maximum: int) -> None:
    if minimum < -(1 << 63) or maximum > (1 << 63) - 1:
        raise ValueError("minimum and maximum must fit in int64")
    if maximum < minimum:
        raise ValueError("maximum must be at least minimum")


def _f64_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return array


def encode_integers(
    values: Iterable[int] | np.ndarray, minimum: int, maximum: int
) -> bytes:
    _bounds(minimum, maximum)
    values_array = _i64(values)
    size = _check(lib().e57_encoded_size(values_array.size, minimum, maximum))
    if not size:
        if values_array.size and np.any(values_array != minimum):
            raise E57CodecError("value outside the declared E57 bounds")
        return b""
    destination = np.empty(size, dtype=np.uint8)
    written = _check(
        lib().e57_encode_i64(
            addr(values_array),
            values_array.size,
            minimum,
            maximum,
            addr(destination),
            size,
        )
    )
    return destination[:written].tobytes()


def decode_integers(
    encoded: bytes | bytearray | memoryview,
    count: int,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be nonnegative")
    _bounds(minimum, maximum)
    destination = np.empty(count, dtype=np.int64)
    if count == 0:
        return destination
    source = np.frombuffer(encoded, dtype=np.uint8)
    _check(
        lib().e57_decode_i64(
            addr(source), source.size, count, minimum, maximum, addr(destination)
        )
    )
    return destination


def encode_scaled(values: Iterable[float] | np.ndarray, codec: ScaledInteger) -> bytes:
    source = _float_array(values, np.float64)
    raw = np.empty(source.size, dtype=np.int64)
    if source.size:
        _check(
            lib().e57_quantize_f64(
                addr(source),
                source.size,
                codec.minimum,
                codec.maximum,
                codec.scale,
                codec.offset,
                addr(raw),
            )
        )
    return encode_integers(raw, codec.minimum, codec.maximum)


def decode_scaled(
    encoded: bytes | bytearray | memoryview, count: int, codec: ScaledInteger
) -> np.ndarray:
    raw = decode_integers(encoded, count, codec.minimum, codec.maximum)
    destination = np.empty(count, dtype=np.float64)
    if count:
        _check(
            lib().e57_dequantize_f64(
                addr(raw), count, codec.scale, codec.offset, addr(destination)
            )
        )
    return destination


def encode_floats(values: Iterable[float] | np.ndarray, precision: int = 64) -> bytes:
    if precision not in (32, 64):
        raise ValueError("precision must be 32 or 64")
    dtype = np.float32 if precision == 32 else np.float64
    source = _float_array(values, dtype)
    destination = np.empty(source.nbytes, dtype=np.uint8)
    if source.size:
        function = lib().e57_encode_f32 if precision == 32 else lib().e57_encode_f64
        _check(function(addr(source), source.size, addr(destination), destination.size))
    return destination.tobytes()


def decode_floats(
    encoded: bytes | bytearray | memoryview, count: int, precision: int = 64
) -> np.ndarray:
    if precision not in (32, 64):
        raise ValueError("precision must be 32 or 64")
    if count < 0:
        raise ValueError("count must be nonnegative")
    dtype = np.float32 if precision == 32 else np.float64
    destination = np.empty(count, dtype=dtype)
    if count:
        source = np.frombuffer(encoded, dtype=np.uint8)
        function = lib().e57_decode_f32 if precision == 32 else lib().e57_decode_f64
        _check(function(addr(source), source.size, count, addr(destination)))
    return destination


def encode_strings(values: Sequence[str]) -> bytes:
    chunks = [value.encode("utf-8") for value in values]
    offsets = np.empty(len(chunks) + 1, dtype=np.int64)
    offsets[0] = 0
    for index, chunk in enumerate(chunks):
        offsets[index + 1] = offsets[index] + len(chunk)
    data = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    sizes = [1 if len(chunk) <= 127 else 8 for chunk in chunks]
    capacity = int(offsets[-1]) + sum(sizes)
    destination = np.empty(capacity, dtype=np.uint8)
    if not chunks:
        return b""
    written = _check(
        lib().e57_encode_strings(
            addr(data),
            data.size,
            addr(offsets),
            len(chunks),
            addr(destination),
            capacity,
        )
    )
    return destination[:written].tobytes()


def decode_strings(
    encoded: bytes | bytearray | memoryview, count: int
) -> list[str]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return []
    source = np.frombuffer(encoded, dtype=np.uint8)
    data = np.empty(source.size, dtype=np.uint8)
    offsets = np.empty(count + 1, dtype=np.int64)
    written = _check(
        lib().e57_decode_strings(
            addr(source), source.size, count, addr(data), data.size, addr(offsets)
        )
    )
    packed = data[:written].tobytes()
    return [
        packed[int(offsets[i]) : int(offsets[i + 1])].decode("utf-8")
        for i in range(count)
    ]


def spherical_to_cartesian(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    source = _f64_points(points)
    destination = np.empty_like(source)
    if source.shape[0]:
        _check(
            lib().e57_spherical_to_cartesian(
                addr(source), source.shape[0], addr(destination)
            )
        )
    return destination


def cartesian_to_spherical(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    source = _f64_points(points)
    destination = np.empty_like(source)
    if source.shape[0]:
        _check(
            lib().e57_cartesian_to_spherical(
                addr(source), source.shape[0], addr(destination)
            )
        )
    return destination


def apply_pose(
    points: Sequence[Sequence[float]] | np.ndarray, pose: ScanPose
) -> np.ndarray:
    source = _f64_points(points)
    destination = np.empty_like(source)
    if source.shape[0]:
        _check(
            lib().e57_apply_pose(
                addr(source),
                source.shape[0],
                addr(destination),
                *pose.rotation,
                *pose.translation,
            )
        )
    return destination


def decode_points(
    streams: Sequence[bytes | bytearray | memoryview],
    count: int,
    codecs: Sequence[ScaledInteger],
    representation: Literal["cartesian", "spherical"] = "cartesian",
    pose: ScanPose | None = None,
) -> np.ndarray:
    """Decode three E57 scaled-integer bytestreams to world-space XYZ points."""
    if len(streams) != 3 or len(codecs) != 3:
        raise ValueError("exactly three streams and codecs are required")
    fields = [
        decode_scaled(stream, count, codec)
        for stream, codec in zip(streams, codecs, strict=True)
    ]
    points = np.column_stack(fields)
    if representation == "spherical":
        points = spherical_to_cartesian(points)
    elif representation != "cartesian":
        raise ValueError("representation must be 'cartesian' or 'spherical'")
    return apply_pose(points, pose) if pose is not None else points


def encode_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    codecs: Sequence[ScaledInteger],
    representation: Literal["cartesian", "spherical"] = "cartesian",
) -> tuple[bytes, bytes, bytes]:
    """Encode XYZ points into three E57 scaled-integer bytestreams."""
    source = _f64_points(points)
    if len(codecs) != 3:
        raise ValueError("exactly three codecs are required")
    if representation == "spherical":
        source = cartesian_to_spherical(source)
    elif representation != "cartesian":
        raise ValueError("representation must be 'cartesian' or 'spherical'")
    encoded = tuple(
        encode_scaled(source[:, index], codecs[index]) for index in range(3)
    )
    return encoded  # type: ignore[return-value]
