"""Independent references transcribed from the corresponding upstream functions."""

from __future__ import annotations

import math
import struct

import numpy as np


def bits_needed(minimum: int, maximum: int) -> int:
    # libE57Format: src/ImageFileImpl.cpp ImageFileImpl::bitsNeeded
    return (maximum - minimum).bit_length()


def encode_integers(values: np.ndarray, minimum: int, maximum: int) -> bytes:
    # libE57Format: src/Encoder.cpp BitpackIntegerEncoder::processRecords
    bits = bits_needed(minimum, maximum)
    if bits == 0:
        if np.any(values != minimum):
            raise ValueError
        return b""
    word_bytes = 1 if bits <= 8 else 2 if bits <= 16 else 4 if bits <= 32 else 8
    word_bits = 8 * word_bytes
    register = 0
    used = 0
    result = bytearray()
    for value in values:
        raw = int(value)
        if raw < minimum or raw > maximum:
            raise ValueError
        unsigned = raw - minimum
        new_used = used + bits
        if new_used > word_bits:
            register |= unsigned << used
            result += (register & ((1 << word_bits) - 1)).to_bytes(
                word_bytes, "little"
            )
            register = unsigned >> (word_bits - used)
            used = new_used - word_bits
        elif new_used == word_bits:
            register |= unsigned << used
            result += register.to_bytes(word_bytes, "little")
            register = 0
            used = 0
        else:
            register |= unsigned << used
            used = new_used
    if used:
        result += register.to_bytes(word_bytes, "little")
    return bytes(result)


def decode_integers(
    encoded: bytes, count: int, minimum: int, maximum: int
) -> np.ndarray:
    # libE57Format: src/Decoder.cpp BitpackIntegerDecoder::inputProcessAligned
    bits = bits_needed(minimum, maximum)
    if bits == 0:
        return np.full(count, minimum, dtype=np.int64)
    value_mask = (1 << bits) - 1
    packed = int.from_bytes(encoded, "little")
    return np.array(
        [minimum + ((packed >> (index * bits)) & value_mask) for index in range(count)],
        dtype=np.int64,
    )


def quantize(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    # libE57Format: src/SourceDestBufferImpl.cpp SourceDestBufferImpl::getNextInt64(scale, offset)
    return np.floor((values - offset) / scale + 0.5).astype(np.int64)


def encode_strings(values: list[str]) -> bytes:
    # libE57Format: src/Encoder.cpp BitpackStringEncoder::processRecords
    result = bytearray()
    for value in values:
        encoded = value.encode()
        prefix = len(encoded) << 1
        result += (
            bytes([prefix])
            if len(encoded) <= 127
            else struct.pack("<Q", prefix | 1)
        )
        result += encoded
    return bytes(result)


def spherical_to_cartesian(points: np.ndarray) -> np.ndarray:
    result = np.empty_like(points)
    horizontal = points[:, 0] * np.cos(points[:, 2])
    result[:, 0] = horizontal * np.cos(points[:, 1])
    result[:, 1] = horizontal * np.sin(points[:, 1])
    result[:, 2] = points[:, 0] * np.sin(points[:, 2])
    return result


def apply_pose(
    points: np.ndarray, rotation: tuple[float, float, float, float], translation
) -> np.ndarray:
    w, x, y, z = rotation
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return points @ matrix.T + np.asarray(translation)
