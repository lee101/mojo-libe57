"""Stable C ABI for the Python ctypes wrapper."""

from e57 import (
    apply_pose,
    bp,
    cartesian_to_spherical,
    decode_f32,
    decode_f64,
    decode_integers,
    decode_strings,
    dequantize_scaled,
    encode_f32,
    encode_f64,
    encode_integers,
    encode_strings,
    encoded_size,
    fp32,
    fp64,
    ip,
    quantize_scaled,
    spherical_to_cartesian,
)


@export("e57_encoded_size")
def e57_encoded_size(n: Int, minimum: Int64, maximum: Int64) abi("C") -> Int:
    if n < 0 or maximum < minimum:
        return -1
    from e57 import bits_needed
    return encoded_size(n, bits_needed(minimum, maximum))


@export("e57_encode_i64")
def e57_encode_i64(
    values: Int,
    n: Int,
    minimum: Int64,
    maximum: Int64,
    destination: Int,
    capacity: Int,
) abi("C") -> Int:
    if n < 0 or capacity < 0 or (n > 0 and values == 0):
        return -1
    if n == 0:
        return 0
    if destination == 0:
        return -1
    return encode_integers(ip(values), n, minimum, maximum, bp(destination), capacity)


@export("e57_decode_i64")
def e57_decode_i64(
    source: Int,
    source_size: Int,
    n: Int,
    minimum: Int64,
    maximum: Int64,
    destination: Int,
) abi("C") -> Int:
    if n < 0 or source_size < 0 or (n > 0 and destination == 0):
        return -1
    if n == 0:
        return 0
    if maximum != minimum and source == 0:
        return -1
    return decode_integers(bp(source), source_size, n, minimum, maximum, ip(destination))


@export("e57_quantize_f64")
def e57_quantize_f64(
    values: Int,
    n: Int,
    minimum: Int64,
    maximum: Int64,
    scale: Float64,
    offset: Float64,
    destination: Int,
) abi("C") -> Int:
    if n < 0 or (n > 0 and (values == 0 or destination == 0)):
        return -1
    if n == 0:
        return 0
    return quantize_scaled(
        fp64(values), n, minimum, maximum, scale, offset, ip(destination)
    )


@export("e57_dequantize_f64")
def e57_dequantize_f64(
    values: Int,
    n: Int,
    scale: Float64,
    offset: Float64,
    destination: Int,
) abi("C") -> Int:
    if n < 0 or (n > 0 and (values == 0 or destination == 0)):
        return -1
    if n > 0:
        dequantize_scaled(ip(values), n, scale, offset, fp64(destination))
    return n


@export("e57_encode_f32")
def e57_encode_f32(
    values: Int, n: Int, destination: Int, capacity: Int
) abi("C") -> Int:
    if n < 0 or capacity < 0 or (n > 0 and (values == 0 or destination == 0)):
        return -1
    if n == 0:
        return 0
    return encode_f32(fp32(values), n, bp(destination), capacity)


@export("e57_decode_f32")
def e57_decode_f32(
    source: Int, source_size: Int, n: Int, destination: Int
) abi("C") -> Int:
    if n < 0 or source_size < 0 or (n > 0 and (source == 0 or destination == 0)):
        return -1
    if n == 0:
        return 0
    return decode_f32(bp(source), source_size, n, fp32(destination))


@export("e57_encode_f64")
def e57_encode_f64(
    values: Int, n: Int, destination: Int, capacity: Int
) abi("C") -> Int:
    if n < 0 or capacity < 0 or (n > 0 and (values == 0 or destination == 0)):
        return -1
    if n == 0:
        return 0
    return encode_f64(fp64(values), n, bp(destination), capacity)


@export("e57_decode_f64")
def e57_decode_f64(
    source: Int, source_size: Int, n: Int, destination: Int
) abi("C") -> Int:
    if n < 0 or source_size < 0 or (n > 0 and (source == 0 or destination == 0)):
        return -1
    if n == 0:
        return 0
    return decode_f64(bp(source), source_size, n, fp64(destination))


@export("e57_encode_strings")
def e57_encode_strings(
    data: Int,
    data_size: Int,
    offsets: Int,
    n: Int,
    destination: Int,
    capacity: Int,
) abi("C") -> Int:
    if n < 0 or data_size < 0 or capacity < 0:
        return -1
    if n > 0 and (offsets == 0 or destination == 0 or (data_size > 0 and data == 0)):
        return -1
    if n == 0:
        return 0
    # A zero-length allocation may have any address, but no data byte is read.
    var data_address = data if data != 0 else destination
    return encode_strings(
        bp(data_address), data_size, ip(offsets), n, bp(destination), capacity
    )


@export("e57_decode_strings")
def e57_decode_strings(
    source: Int,
    source_size: Int,
    n: Int,
    data: Int,
    data_capacity: Int,
    offsets: Int,
) abi("C") -> Int:
    if n < 0 or source_size < 0 or data_capacity < 0:
        return -1
    if n > 0 and (source == 0 or data == 0 or offsets == 0):
        return -1
    if n == 0:
        return 0
    return decode_strings(
        bp(source), source_size, n, bp(data), data_capacity, ip(offsets)
    )


@export("e57_spherical_to_cartesian")
def e57_spherical_to_cartesian(
    source: Int, n: Int, destination: Int
) abi("C") -> Int:
    if n < 0 or (n > 0 and (source == 0 or destination == 0)):
        return -1
    if n > 0:
        spherical_to_cartesian(fp64(source), n, fp64(destination))
    return n


@export("e57_cartesian_to_spherical")
def e57_cartesian_to_spherical(
    source: Int, n: Int, destination: Int
) abi("C") -> Int:
    if n < 0 or (n > 0 and (source == 0 or destination == 0)):
        return -1
    if n > 0:
        cartesian_to_spherical(fp64(source), n, fp64(destination))
    return n


@export("e57_apply_pose")
def e57_apply_pose(
    source: Int,
    n: Int,
    destination: Int,
    w: Float64,
    qx: Float64,
    qy: Float64,
    qz: Float64,
    tx: Float64,
    ty: Float64,
    tz: Float64,
) abi("C") -> Int:
    if n < 0 or (n > 0 and (source == 0 or destination == 0)):
        return -1
    if n > 0:
        apply_pose(
            fp64(source), n, fp64(destination), w, qx, qy, qz, tx, ty, tz
        )
    return n
