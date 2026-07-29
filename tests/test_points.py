from __future__ import annotations

import numpy as np

import mojo_libe57 as e57

from reference import apply_pose as reference_apply_pose
from reference import spherical_to_cartesian as reference_spherical_to_cartesian


def test_spherical_cartesian_matches_reference():
    spherical = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, np.pi / 2, 0.0],
            [3.0, -np.pi, np.pi / 4],
            [4.0, 1.25, -np.pi / 2],
        ]
    )
    np.testing.assert_allclose(
        e57.spherical_to_cartesian(spherical),
        reference_spherical_to_cartesian(spherical),
        rtol=2e-16,
        atol=2e-16,
    )


def test_spherical_cartesian_simd_tail():
    rng = np.random.default_rng(5701)
    spherical = np.empty((17, 3), dtype=np.float64)
    spherical[:, 0] = rng.uniform(0.0, 200.0, len(spherical))
    spherical[:, 1] = rng.uniform(-np.pi, np.pi, len(spherical))
    spherical[:, 2] = rng.uniform(-np.pi / 2, np.pi / 2, len(spherical))
    np.testing.assert_allclose(
        e57.spherical_to_cartesian(spherical),
        reference_spherical_to_cartesian(spherical),
        rtol=3e-15,
        atol=3e-14,
    )


def test_point_inputs_are_copied_to_contiguous_float64_storage():
    base = np.arange(30, dtype=np.float32).reshape(10, 3)
    noncontiguous = base[::2]
    result = e57.apply_pose(noncontiguous, e57.ScanPose())
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, noncontiguous)


def test_spherical_cartesian_parallel_threshold():
    rng = np.random.default_rng(5702)
    for count in (65_535, 65_536):
        spherical = np.empty((count, 3), dtype=np.float64)
        spherical[:, 0] = rng.uniform(0.0, 200.0, count)
        spherical[:, 1] = rng.uniform(-np.pi, np.pi, count)
        spherical[:, 2] = rng.uniform(-np.pi / 2, np.pi / 2, count)
        np.testing.assert_allclose(
            e57.spherical_to_cartesian(spherical),
            reference_spherical_to_cartesian(spherical),
            rtol=3e-15,
            atol=3e-14,
        )


def test_cartesian_spherical_roundtrip_degenerate_points():
    cartesian = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, -3.0],
        ]
    )
    spherical = e57.cartesian_to_spherical(cartesian)
    np.testing.assert_allclose(
        e57.spherical_to_cartesian(spherical), cartesian, rtol=1e-15, atol=4e-16
    )
    np.testing.assert_array_equal(spherical[0], [0.0, 0.0, 0.0])


def test_pose_matches_unit_quaternion_reference():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    angle = 0.73
    rotation = (
        np.cos(angle / 2),
        np.sin(angle / 2) / np.sqrt(3),
        np.sin(angle / 2) / np.sqrt(3),
        np.sin(angle / 2) / np.sqrt(3),
    )
    translation = (10.0, -2.0, 0.25)
    pose = e57.ScanPose(rotation, translation)
    np.testing.assert_allclose(
        e57.apply_pose(points, pose),
        reference_apply_pose(points, rotation, translation),
        rtol=3e-16,
        atol=2e-15,
    )


def test_pose_identity_empty_and_duplicate_points():
    empty = np.empty((0, 3))
    assert e57.apply_pose(empty, e57.ScanPose()).shape == (0, 3)
    points = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(e57.apply_pose(points, e57.ScanPose()), points)


def test_encode_decode_cartesian_points():
    points = np.array(
        [[-1.25, 2.5, 0.0], [3.75, -4.0, 8.125], [-1.25, 2.5, 0.0]]
    )
    codecs = (
        e57.ScaledInteger(-200, 400, 0.01),
        e57.ScaledInteger(-500, 300, 0.01),
        e57.ScaledInteger(0, 1000, 0.01),
    )
    streams = e57.encode_points(points, codecs)
    expected = points.copy()
    expected[1, 2] = 8.13
    np.testing.assert_allclose(
        e57.decode_points(streams, len(points), codecs), expected, rtol=0, atol=0
    )


def test_decode_spherical_points_and_pose():
    spherical = np.array([[2.0, 0.0, 0.0], [4.0, np.pi / 2, 0.0]])
    codecs = (
        e57.ScaledInteger(0, 1000, 0.01),
        e57.ScaledInteger(-4000, 4000, 0.001),
        e57.ScaledInteger(-2000, 2000, 0.001),
    )
    streams = tuple(
        e57.encode_scaled(spherical[:, index], codecs[index]) for index in range(3)
    )
    quantized = np.column_stack(
        [e57.decode_scaled(streams[i], 2, codecs[i]) for i in range(3)]
    )
    pose = e57.ScanPose(translation=(1.0, 2.0, 3.0))
    expected = reference_spherical_to_cartesian(quantized) + pose.translation
    np.testing.assert_allclose(
        e57.decode_points(streams, 2, codecs, "spherical", pose),
        expected,
        rtol=2e-16,
        atol=2e-16,
    )
