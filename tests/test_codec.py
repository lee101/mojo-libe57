from __future__ import annotations

import numpy as np
import pytest

import mojo_libe57 as e57
from mojo_libe57._lib import lib

from reference import encode_integers as reference_encode
from reference import encode_strings as reference_encode_strings
from reference import quantize as reference_quantize


@pytest.mark.parametrize(
    ("minimum", "maximum", "values"),
    [
        (7, 7, [7, 7, 7]),
        (0, 1, [0, 1, 1, 0, 1, 0, 0, 1]),
        (-5, 5, [-5, 5, 0, -1, 2]),
        (-1000, 1000, [-1000, 1000, 0, 999, -999]),
        (-(1 << 31), (1 << 31) - 1, [-(1 << 31), 0, (1 << 31) - 1]),
        (-(1 << 63), (1 << 63) - 1, [-(1 << 63), -1, 0, (1 << 63) - 1]),
    ],
)
def test_integer_codec_matches_upstream_algorithm(minimum, maximum, values):
    source = np.array(values, dtype=np.int64)
    expected = reference_encode(source, minimum, maximum)
    encoded = e57.encode_integers(source, minimum, maximum)
    assert encoded == expected
    np.testing.assert_array_equal(
        e57.decode_integers(encoded, len(values), minimum, maximum), source
    )


def test_fixed_upstream_bit_order_fixture():
    # Four-bit values 0, 10, 5 pack least-significant bit first, padded to bytes.
    assert e57.encode_integers([0, 10, 5], 0, 10) == bytes.fromhex("a005")


@pytest.mark.parametrize("width", range(65))
def test_every_claimed_integer_width(width):
    if width == 0:
        minimum, maximum = -7, -7
    elif width == 64:
        minimum, maximum = -(1 << 63), (1 << 63) - 1
    else:
        minimum, maximum = 0, (1 << width) - 1
    values = np.array([minimum, maximum, minimum], dtype=np.int64)
    encoded = e57.encode_integers(values, minimum, maximum)
    assert encoded == reference_encode(values, minimum, maximum)
    np.testing.assert_array_equal(
        e57.decode_integers(encoded, len(values), minimum, maximum), values
    )


def test_empty_and_constant_streams():
    assert e57.encode_integers([], -4, 11) == b""
    assert e57.decode_integers(b"", 0, -4, 11).shape == (0,)
    assert e57.encode_integers([12, 12], 12, 12) == b""
    np.testing.assert_array_equal(
        e57.decode_integers(b"", 2, 12, 12), np.array([12, 12])
    )


def test_codec_rejects_out_of_bounds_and_truncation():
    with pytest.raises(e57.E57CodecError, match="outside"):
        e57.encode_integers([0, 11], 0, 10)
    with pytest.raises(e57.E57CodecError, match="truncated"):
        e57.decode_integers(b"\x00", 10, 0, 255)


def test_integer_codec_rejects_silent_narrowing_and_invalid_shapes():
    with pytest.raises(TypeError, match="integer"):
        e57.encode_integers([1.5], 0, 2)
    with pytest.raises(OverflowError, match="int64"):
        e57.encode_integers(np.array([1 << 63], dtype=np.uint64), 0, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        e57.encode_integers([[1, 2]], 0, 2)
    with pytest.raises(ValueError, match="at least"):
        e57.decode_integers(b"", 0, 1, 0)


@pytest.mark.parametrize(
    ("scale", "offset", "values"),
    [
        (0.01, -3.0, [-3.0, -2.995, 0.0, 4.499]),
        (2.0, 1.0, [-4.0, 0.0, 2.0, 8.0]),
        (-0.5, 3.0, [3.0, 2.75, 1.0, -2.0]),
    ],
)
def test_scaled_integer_rounding_matches_source(scale, offset, values):
    source = np.array(values)
    raw = reference_quantize(source, scale, offset)
    codec = e57.ScaledInteger(int(raw.min()), int(raw.max()), scale, offset)
    encoded = e57.encode_scaled(source, codec)
    np.testing.assert_array_equal(
        e57.decode_integers(encoded, len(source), codec.minimum, codec.maximum), raw
    )
    np.testing.assert_allclose(
        e57.decode_scaled(encoded, len(source), codec),
        raw.astype(float) * scale + offset,
        rtol=0,
        atol=1e-16,
    )


def test_scaled_half_ties_round_toward_positive_infinity():
    codec = e57.ScaledInteger(-2, 2)
    encoded = e57.encode_scaled(np.array([-1.5, -0.5, 0.5, 1.5]), codec)
    np.testing.assert_array_equal(
        e57.decode_integers(encoded, 4, -2, 2), [-1, 0, 1, 2]
    )


def test_scaled_rejects_nonfinite_and_zero_scale():
    with pytest.raises(ValueError, match="nonzero"):
        e57.ScaledInteger(0, 1, 0.0)
    with pytest.raises(e57.E57CodecError, match="representable"):
        e57.encode_scaled([np.nan], e57.ScaledInteger(0, 1))
    with pytest.raises(e57.E57CodecError, match="representable"):
        e57.encode_scaled([np.inf], e57.ScaledInteger(0, 1))
    with pytest.raises(ValueError, match="finite"):
        e57.ScaledInteger(0, 1, np.inf)


@pytest.mark.parametrize("precision", [32, 64])
def test_float_codec_is_bit_preserving(precision):
    dtype = np.float32 if precision == 32 else np.float64
    values = np.array([0.0, -0.0, 1.5, -np.inf, np.inf, np.nan], dtype=dtype)
    encoded = e57.encode_floats(values, precision)
    assert encoded == values.tobytes()
    decoded = e57.decode_floats(encoded, len(values), precision)
    assert decoded.view(f"u{precision // 8}").tobytes() == values.view(
        f"u{precision // 8}"
    ).tobytes()


def test_float_codec_rejects_truncation():
    with pytest.raises(e57.E57CodecError, match="truncated"):
        e57.decode_floats(b"\0" * 7, 1, 64)
    with pytest.raises(ValueError, match="nonnegative"):
        e57.decode_floats(b"", -1, 64)


def test_string_codec_short_long_empty_and_utf8():
    values = ["", "ascii", "München", "x" * 127, "λ" * 64]
    encoded = e57.encode_strings(values)
    assert encoded == reference_encode_strings(values)
    assert e57.decode_strings(encoded, len(values)) == values
    assert encoded[len(reference_encode_strings(values[:4]))] & 1 == 1


def test_string_codec_empty_and_truncated():
    assert e57.encode_strings([]) == b""
    assert e57.decode_strings(b"", 0) == []
    with pytest.raises(e57.E57CodecError, match="truncated"):
        e57.decode_strings(b"\x0aabc", 1)


def test_c_abi_rejects_invalid_counts_and_null_pointers():
    library = lib()
    assert library.e57_encode_i64(0, 1, 0, 1, 0, 0) == -1
    assert library.e57_decode_f64(0, 8, 1, 0) == -1
    assert library.e57_spherical_to_cartesian(0, 1, 0) == -1
    assert library.e57_apply_pose(0, -1, 0, *([0.0] * 7)) == -1
