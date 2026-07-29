from __future__ import annotations

import numpy as np
import pye57

import mojo_libe57 as e57


def test_real_libe57format_file_roundtrip_and_scan_pose(tmp_path):
    path = tmp_path / "parity.e57"
    local = np.array(
        [
            [-1.25, 2.5, 0.0],
            [3.75, -4.0, 8.125],
            [0.0, 0.0, 0.0],
            [-1.25, 2.5, 0.0],
        ],
        dtype=np.float64,
    )
    angle = 0.61
    rotation = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    translation = np.array([10.0, -2.0, 0.25])
    colors = np.array([0, 1, 127, 255], dtype=np.uint8)

    writer = pye57.E57(str(path), mode="w")
    writer.write_scan_raw(
        {
            "cartesianX": local[:, 0],
            "cartesianY": local[:, 1],
            "cartesianZ": local[:, 2],
            "colorRed": colors,
            "colorGreen": 255 - colors,
            "colorBlue": colors,
        },
        rotation=rotation,
        translation=translation,
    )
    writer.close()

    reader = pye57.E57(str(path))
    raw = reader.read_scan_raw(0)
    transformed = reader.read_scan(
        0, colors=True, transform=True, ignore_missing_fields=True
    )
    header = reader.get_header(0)
    header_rotation = header.rotation.copy()
    header_translation = header.translation.copy()
    reader.close()

    # pye57 writes coordinates through libE57Format's single-precision codec.
    np.testing.assert_array_equal(raw["cartesianX"], local[:, 0].astype(np.float32))
    np.testing.assert_array_equal(raw["colorRed"], colors)
    np.testing.assert_allclose(header_rotation, rotation, rtol=0, atol=0)
    np.testing.assert_allclose(header_translation, translation, rtol=0, atol=0)

    mojo_world = e57.apply_pose(
        local.astype(np.float32).astype(np.float64),
        e57.ScanPose(tuple(header_rotation), tuple(header_translation)),
    )
    reference_world = np.column_stack(
        [
            transformed["cartesianX"],
            transformed["cartesianY"],
            transformed["cartesianZ"],
        ]
    )
    np.testing.assert_allclose(mojo_world, reference_world, rtol=2e-15, atol=2e-15)
