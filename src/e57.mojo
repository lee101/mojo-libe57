"""Compute kernels ported from libE57Format's compressed-vector path."""

from std.math import atan2, cos, floor, sin, sqrt
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime SPHERICAL_PARALLEL_POINTS = 65536
comptime SPHERICAL_MAX_WORKERS = 32


@always_inline
def bp(address: Int) -> BPtr:
    return BPtr(unsafe_from_address=address)


@always_inline
def ip(address: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=address)


@always_inline
def fp32(address: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=address)


@always_inline
def fp64(address: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=address)


# libE57Format: src/ImageFileImpl.cpp ImageFileImpl::bitsNeeded
@always_inline
def bits_needed(minimum: Int64, maximum: Int64) -> Int:
    var states = UInt64(maximum) - UInt64(minimum)
    var count = 0
    if states & UInt64(0xFFFFFFFF00000000):
        states >>= 32
        count += 32
    if states & UInt64(0xFFFF0000):
        states >>= 16
        count += 16
    if states & UInt64(0xFF00):
        states >>= 8
        count += 8
    if states & UInt64(0xF0):
        states >>= 4
        count += 4
    if states & UInt64(0xC):
        states >>= 2
        count += 2
    if states & UInt64(0x2):
        states >>= 1
        count += 1
    if states & UInt64(1):
        count += 1
    return count


@always_inline
def register_bytes(bits: Int) -> Int:
    if bits <= 8:
        return 1
    if bits <= 16:
        return 2
    if bits <= 32:
        return 4
    return 8


@always_inline
def encoded_size(n: Int, bits: Int) -> Int:
    if bits == 0 or n == 0:
        return 0
    var word = register_bytes(bits)
    var word_bits = word * 8
    if n < 0 or n > (9223372036854775807 - (word_bits - 1)) // bits:
        return -1
    return ((n * bits + word_bits - 1) // word_bits) * word


# libE57Format: src/Encoder.cpp BitpackIntegerEncoder<RegisterT>::processRecords
def encode_integers(
    values: I64Ptr,
    n: Int,
    minimum: Int64,
    maximum: Int64,
    destination: BPtr,
    capacity: Int,
) -> Int:
    if n < 0 or maximum < minimum:
        return -1
    var bits = bits_needed(minimum, maximum)
    var required = encoded_size(n, bits)
    if required < 0:
        return -1
    if capacity < required:
        return -2
    for j in range(required):
        destination[j] = UInt8(0)
    if bits == 0:
        for i in range(n):
            if values[i] != minimum:
                return -3
        return 0

    var bit_position = 0
    for i in range(n):
        var value = values[i]
        if value < minimum or value > maximum:
            return -3
        var unsigned_value = UInt64(value) - UInt64(minimum)
        var remaining = bits
        var shift = 0
        while remaining > 0:
            var byte_index = bit_position >> 3
            var bit_offset = bit_position & 7
            var take = min(remaining, 8 - bit_offset)
            var mask = (UInt64(1) << UInt64(take)) - UInt64(1)
            destination[byte_index] |= UInt8(
                (unsigned_value >> UInt64(shift) & mask) << UInt64(bit_offset)
            )
            bit_position += take
            shift += take
            remaining -= take
    return required


# libE57Format: src/Decoder.cpp BitpackIntegerDecoder<RegisterT>::inputProcessAligned
def decode_integers(
    source: BPtr,
    source_size: Int,
    n: Int,
    minimum: Int64,
    maximum: Int64,
    destination: I64Ptr,
) -> Int:
    if n < 0 or maximum < minimum:
        return -1
    var bits = bits_needed(minimum, maximum)
    if bits == 0:
        for i in range(n):
            destination[i] = minimum
        return n
    var required = encoded_size(n, bits)
    if required < 0:
        return -1
    if source_size < required:
        return -2

    var bit_position = 0
    for i in range(n):
        var unsigned_value = UInt64(0)
        var remaining = bits
        var shift = 0
        while remaining > 0:
            var byte_index = bit_position >> 3
            var bit_offset = bit_position & 7
            var take = min(remaining, 8 - bit_offset)
            var mask = (UInt64(1) << UInt64(take)) - UInt64(1)
            unsigned_value |= (
                UInt64(source[byte_index]) >> UInt64(bit_offset) & mask
            ) << UInt64(shift)
            bit_position += take
            shift += take
            remaining -= take
        destination[i] = Int64(UInt64(minimum) + unsigned_value)
    return n


# libE57Format: src/SourceDestBufferImpl.cpp SourceDestBufferImpl::getNextInt64(scale, offset)
def quantize_scaled(
    values: F64Ptr,
    n: Int,
    minimum: Int64,
    maximum: Int64,
    scale: Float64,
    offset: Float64,
    destination: I64Ptr,
) -> Int:
    if n < 0 or scale == 0.0 or maximum < minimum:
        return -1
    for i in range(n):
        var value = (values[i] - offset) / scale
        if value != value or value < -9223372036854775808.0 or value >= 9223372036854775808.0:
            return -4
        var raw = Int64(floor(value + 0.5))
        if raw < minimum or raw > maximum:
            return -3
        destination[i] = raw
    return n


# libE57Format: src/SourceDestBufferImpl.cpp SourceDestBufferImpl::setNextInt64(scale, offset)
def dequantize_scaled(
    values: I64Ptr,
    n: Int,
    scale: Float64,
    offset: Float64,
    destination: F64Ptr,
):
    for i in range(n):
        destination[i] = Float64(values[i]) * scale + offset


# libE57Format: src/Encoder.cpp BitpackFloatEncoder::processRecords
def encode_f32(values: F32Ptr, n: Int, destination: BPtr, capacity: Int) -> Int:
    if n < 0 or n > 9223372036854775807 // 4:
        return -1
    var required = n * 4
    if capacity < required:
        return -2
    var src = values.bitcast[UInt8]()
    for i in range(required):
        destination[i] = src[i]
    return required


# libE57Format: src/Decoder.cpp BitpackFloatDecoder::inputProcessAligned
def decode_f32(source: BPtr, source_size: Int, n: Int, destination: F32Ptr) -> Int:
    if n < 0 or n > 9223372036854775807 // 4:
        return -1
    var required = n * 4
    if source_size < required:
        return -2
    var dest = destination.bitcast[UInt8]()
    for i in range(required):
        dest[i] = source[i]
    return n


# libE57Format: src/Encoder.cpp BitpackFloatEncoder::processRecords
def encode_f64(values: F64Ptr, n: Int, destination: BPtr, capacity: Int) -> Int:
    if n < 0 or n > 9223372036854775807 // 8:
        return -1
    var required = n * 8
    if capacity < required:
        return -2
    var src = values.bitcast[UInt8]()
    for i in range(required):
        destination[i] = src[i]
    return required


# libE57Format: src/Decoder.cpp BitpackFloatDecoder::inputProcessAligned
def decode_f64(source: BPtr, source_size: Int, n: Int, destination: F64Ptr) -> Int:
    if n < 0 or n > 9223372036854775807 // 8:
        return -1
    var required = n * 8
    if source_size < required:
        return -2
    var dest = destination.bitcast[UInt8]()
    for i in range(required):
        dest[i] = source[i]
    return n


# libE57Format: src/Encoder.cpp BitpackStringEncoder::processRecords
def encode_strings(
    data: BPtr,
    data_size: Int,
    offsets: I64Ptr,
    n: Int,
    destination: BPtr,
    capacity: Int,
) -> Int:
    if n < 0 or data_size < 0 or capacity < 0:
        return -1
    var write = 0
    for i in range(n):
        var start = offsets[i]
        var end = offsets[i + 1]
        if start < 0 or end < start or end > Int64(data_size):
            return -1
        var length = end - start
        var prefix = 1 if length <= 127 else 8
        if write > capacity - prefix or Int(length) > capacity - write - prefix:
            return -2
        if prefix == 1:
            destination[write] = UInt8(UInt64(length) << 1)
            write += 1
        else:
            var length_prefix = (UInt64(length) << 1) | UInt64(1)
            for j in range(8):
                destination[write + j] = UInt8(length_prefix >> UInt64(j * 8))
            write += 8
        for j in range(Int(length)):
            destination[write + j] = data[Int(start) + j]
        write += Int(length)
    return write


# libE57Format: src/Decoder.cpp BitpackStringDecoder::inputProcessAligned
def decode_strings(
    source: BPtr,
    source_size: Int,
    n: Int,
    data: BPtr,
    data_capacity: Int,
    offsets: I64Ptr,
) -> Int:
    if n < 0:
        return -1
    var read = 0
    var write = 0
    offsets[0] = 0
    for i in range(n):
        if read >= source_size:
            return -2
        var first = source[read]
        var length = UInt64(0)
        if first & UInt8(1):
            if read + 8 > source_size:
                return -2
            var prefix = UInt64(0)
            for j in range(8):
                prefix |= UInt64(source[read + j]) << UInt64(j * 8)
            length = prefix >> 1
            read += 8
        else:
            length = UInt64(first) >> 1
            read += 1
        if length > UInt64(source_size - read):
            return -2
        if length > UInt64(data_capacity - write):
            return -3
        for j in range(Int(length)):
            data[write + j] = source[read + j]
        read += Int(length)
        write += Int(length)
        offsets[i + 1] = Int64(write)
    return write


# ASTM E57 coordinates used by libE57Format: include/E57SimpleData.h SphericalBounds
def spherical_to_cartesian(source: F64Ptr, n: Int, destination: F64Ptr):
    comptime W = simdwidthof[DType.float64]()
    var workers = 1
    if n >= SPHERICAL_PARALLEL_POINTS:
        workers = min(n, min(num_physical_cores(), SPHERICAL_MAX_WORKERS))

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var stop = (worker + 1) * n // workers
        var vector_stop = start + (stop - start) // W * W
        for i in range(start, vector_stop, W):
            var radius = (source + 3 * i).strided_load[width=W](3)
            var azimuth = (source + 3 * i + 1).strided_load[width=W](3)
            var elevation = (source + 3 * i + 2).strided_load[width=W](3)
            var horizontal = radius * cos(elevation)
            (destination + 3 * i).strided_store[width=W](
                horizontal * cos(azimuth), 3
            )
            (destination + 3 * i + 1).strided_store[width=W](
                horizontal * sin(azimuth), 3
            )
            (destination + 3 * i + 2).strided_store[width=W](
                radius * sin(elevation), 3
            )
        for i in range(vector_stop, stop):
            var radius = source[3 * i]
            var azimuth = source[3 * i + 1]
            var elevation = source[3 * i + 2]
            var horizontal = radius * cos(elevation)
            destination[3 * i] = horizontal * cos(azimuth)
            destination[3 * i + 1] = horizontal * sin(azimuth)
            destination[3 * i + 2] = radius * sin(elevation)

    for worker in range(workers):
        process(worker)


# ASTM E57 coordinates used by libE57Format: include/E57SimpleData.h CartesianBounds
def cartesian_to_spherical(source: F64Ptr, n: Int, destination: F64Ptr):
    for i in range(n):
        var x = source[3 * i]
        var y = source[3 * i + 1]
        var z = source[3 * i + 2]
        var horizontal = sqrt(x * x + y * y)
        destination[3 * i] = sqrt(horizontal * horizontal + z * z)
        destination[3 * i + 1] = atan2(y, x)
        destination[3 * i + 2] = atan2(z, horizontal)


# libE57Format: include/E57SimpleData.h RigidBodyTransform; src/ReaderImpl.cpp ReaderImpl::ReadData3D
def apply_pose(
    source: F64Ptr,
    n: Int,
    destination: F64Ptr,
    w: Float64,
    qx: Float64,
    qy: Float64,
    qz: Float64,
    tx: Float64,
    ty: Float64,
    tz: Float64,
):
    # E57 declares a unit quaternion; preserve upstream values without normalization.
    var xx = qx * qx
    var yy = qy * qy
    var zz = qz * qz
    var xy = qx * qy
    var xz = qx * qz
    var yz = qy * qz
    var wx = w * qx
    var wy = w * qy
    var wz = w * qz
    var r00 = 1.0 - 2.0 * (yy + zz)
    var r01 = 2.0 * (xy - wz)
    var r02 = 2.0 * (xz + wy)
    var r10 = 2.0 * (xy + wz)
    var r11 = 1.0 - 2.0 * (xx + zz)
    var r12 = 2.0 * (yz - wx)
    var r20 = 2.0 * (xz - wy)
    var r21 = 2.0 * (yz + wx)
    var r22 = 1.0 - 2.0 * (xx + yy)
    for i in range(n):
        var x = source[3 * i]
        var y = source[3 * i + 1]
        var z = source[3 * i + 2]
        destination[3 * i] = r00 * x + r01 * y + r02 * z + tx
        destination[3 * i + 1] = r10 * x + r11 * y + r12 * z + ty
        destination[3 * i + 2] = r20 * x + r21 * y + r22 * z + tz
