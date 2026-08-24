# mojo-libe57

`mojo-libe57` ports the compute-heavy compressed-vector and point kernels from
[libE57Format](https://github.com/asmaloney/libE57Format) to Mojo, with a
NumPy-friendly Python API. It is useful when an E57 reader or writer already
has the field metadata and compressed bytestreams and needs fast bulk codec,
coordinate, or scan-pose work.

The port was made from upstream revision
`2b84ff0d4ee451d9c191a5eab48a889db8823cc6`, not from a format summary.
Upstream is distributed under the
[Boost Software License 1.0](https://github.com/asmaloney/libE57Format/blob/master/LICENSE.md).
This repository's original glue and API are MIT licensed; the source-derived
portions remain subject to the upstream terms recorded in [NOTICE](NOTICE).

## Coverage

Included:

- E57 integer bytestream encoding and decoding for widths from 0 through 64
  bits, including upstream's 1/2/4/8-byte register padding
- scaled-integer quantization and expansion, including the upstream
  `floor(x + 0.5)` tie behavior, arbitrary scale, and offset
- bit-preserving single- and double-precision float bytestreams
- short and long UTF-8 string prefixes
- batched Cartesian/spherical point conversion
- batched unit-quaternion scan pose application
- three-field scaled point encode/decode helpers

Not included:

- XML metadata parsing or serialization
- E57 page checksums, logical/physical offset mapping, file headers, blobs, or
  image payloads
- compressed-vector packet scheduling, index packets, or random access
- a complete `.e57` file API

Those excluded paths are primarily I/O and object-tree control flow. For
complete file handling, use libE57Format or `pye57` and hand its arrays to
these kernels.

## Install

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-libe57.so`. All supported workflows run
inside the pinned Pixi environment:

```bash
pixi run test
pixi run bench
```

## Usage

```python
import numpy as np
from mojo_libe57 import ScaledInteger, ScanPose, decode_points, encode_points

points = np.array([
    [-1.25, 2.50, 0.00],
    [ 3.75, -4.00, 8.13],
], dtype=np.float64)

codecs = (
    ScaledInteger(-200, 400, scale=0.01),
    ScaledInteger(-500, 300, scale=0.01),
    ScaledInteger(0, 1000, scale=0.01),
)

streams = encode_points(points, codecs)
world = decode_points(
    streams,
    count=len(points),
    codecs=codecs,
    pose=ScanPose(translation=(10.0, 0.0, 0.0)),
)
np.testing.assert_allclose(world, points + [10.0, 0.0, 0.0])
```

Save that as `example.py` and run it with `pixi run python example.py`.
Raw helpers such as `encode_integers`, `decode_scaled`, `encode_floats`, and
`encode_strings` are also exported.

## How it works

Python owns every allocation. Contiguous NumPy buffers cross a small `ctypes`
C ABI as integer addresses, and the Mojo library rebuilds typed
`UnsafePointer` values internally. No Mojo allocation or ownership crosses
the boundary.

Integer fields use separate byte bytestreams, exactly as E57 compressed-vector
channels do. Values are stored as the unsigned difference from the declared
minimum, least-significant bit first. Point arrays are interleaved
row-major `(n, 3)` float64 buffers; point fields are encoded into three
structure-of-arrays bytestreams. Float streams use the host's native
little-endian representation, matching the supported Linux x86-64 target.

Spherical conversion uses native-width float64 SIMD with strided loads and
stores over the interleaved point rows, followed by a scalar remainder. Inputs
of at least 65,536 points are divided across up to 32 physical-core workers;
smaller inputs stay serial to avoid thread-launch overhead. Contiguous float64
NumPy inputs remain zero-copy through the CPU FFI boundary.

Spherical conversion is the only kernel here with enough arithmetic intensity
to plausibly benefit from a GPU. A GPU path was evaluated with 13,533 MiB free,
but the pinned Mojo toolchain does not support float64 `sin` on NVIDIA GPUs.
Using float32 would violate the existing parity tolerances, so no GPU path is
exposed.

Tests use `pye57`, which binds libE57Format, for a real file write/read and
pose parity test. The binding does not expose raw compressed bytestreams, so
codec byte parity uses an independent NumPy/Python reference transcribed from
upstream `Encoder.cpp`, `Decoder.cpp`, `ImageFileImpl.cpp`, and
`SourceDestBufferImpl.cpp`, plus fixed byte fixtures and round-trip
invariants.

## Benchmarks

Measured by `pixi run bench` on 2026-08-24 on an Intel Xeon E5-2697 v4 at
2.30 GHz, 72 logical CPUs, Linux 6.8.0-136-generic. Each row uses 1,000,000
records or points and reports the median of seven runs.

| Kernel | Mojo ms | Reference | Reference ms | Speedup |
|---|---:|---|---:|---:|
| 12-bit integer decode | 4.509 | source-derived NumPy | 56.336 | 12.49x |
| spherical to Cartesian | 10.627 | NumPy | 104.638 | 9.85x |
| scan pose | 5.903 | pye57/NumPy | 19.639 | 3.33x |

The locked pre-optimization run on the same machine measured Mojo times of
4.756 ms, 77.670 ms, and 5.272 ms respectively. The spherical kernel improved
by 7.31x; the untouched codec and pose rows vary with concurrent system load.

The benchmark performs parity assertions before timing and prints its machine
description with the table.
