import numpy as np
import pytest

from radar_speed_estimate import (
    EstimationError,
    estimate_velocity_fused,
    estimate_velocity_from_positions,
    estimate_velocity_from_radial,
    estimate_velocity_from_time_xyvr,
    estimate_velocity_from_xyvr,
    radial_consistency_error,
)


def test_radial_two_measurements_exact_solution() -> None:
    v_true = np.array([3.0, -1.0])
    positions = np.array([[10.0, 0.0], [0.0, 10.0]])
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    vr = unit @ v_true

    est = estimate_velocity_from_radial(positions, vr)

    np.testing.assert_allclose(est.vector, v_true, atol=1e-12)
    assert est.rank == 2


def test_radial_overdetermined_solution_noise_free() -> None:
    v_true = np.array([1.2, 2.3])
    positions = np.array(
        [[10.0, 1.0], [7.0, 7.0], [2.0, 12.0], [-5.0, 9.0], [-9.0, 3.0]]
    )
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    vr = unit @ v_true

    est = estimate_velocity_from_radial(positions, vr)

    np.testing.assert_allclose(est.vector, v_true, atol=1e-12)
    np.testing.assert_allclose(est.residual_l2, 0.0, atol=1e-12)


def test_radial_degenerate_geometry_raises() -> None:
    positions = np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    vr = np.array([1.0, 1.0, 1.0])

    with pytest.raises(EstimationError, match=r"rank\(U\) < 2"):
        estimate_velocity_from_radial(positions, vr)


def test_position_based_exact_constant_velocity() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    p0 = np.array([5.0, -2.0])
    v_true = np.array([0.8, 1.5])
    positions = p0 + t[:, None] * v_true

    est = estimate_velocity_from_positions(t, positions)

    np.testing.assert_allclose(est.vector, v_true, atol=1e-12)
    np.testing.assert_allclose(est.residual_l2, 0.0, atol=1e-12)


def test_position_and_radial_consistency() -> None:
    t = np.array([0.0, 0.5, 1.2, 2.0, 2.5])
    p0 = np.array([15.0, -6.0])
    v_true = np.array([-2.2, 0.9])
    positions = p0 + t[:, None] * v_true
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    vr = unit @ v_true

    est = estimate_velocity_from_positions(t, positions)
    err = radial_consistency_error(est.vector, positions, vr)

    np.testing.assert_allclose(err, np.zeros_like(err), atol=1e-12)


def test_randomized_stability_100_trials() -> None:
    rng = np.random.default_rng(42)
    for _ in range(100):
        v_true = rng.uniform(-5.0, 5.0, size=2)
        angles = np.sort(rng.uniform(-2.7, 2.7, size=8))
        radius = rng.uniform(5.0, 30.0, size=8)
        positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

        unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        vr = unit @ v_true
        est = estimate_velocity_from_radial(positions, vr)

        np.testing.assert_allclose(est.vector, v_true, atol=1e-10)


def test_noisy_many_measurements_and_fused_estimator() -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 5.0, 40)
    p0 = np.array([20.0, -9.0])
    v_true = np.array([-3.2, 1.7])

    positions_true = p0 + t[:, None] * v_true
    los = positions_true / np.linalg.norm(positions_true, axis=1, keepdims=True)
    vr_true = los @ v_true

    # Add moderate artificial noise.
    positions_noisy = positions_true + rng.normal(0.0, 0.35, size=positions_true.shape)
    vr_noisy = vr_true + rng.normal(0.0, 0.12, size=vr_true.shape)

    est_radial = estimate_velocity_from_radial(positions_noisy, vr_noisy)
    est_pos = estimate_velocity_from_positions(t, positions_noisy)
    est_fused = estimate_velocity_fused(
        t,
        positions_noisy,
        vr_noisy,
        weight_position=1.0 / (0.35**2),
        weight_radial=1.0 / (0.12**2),
    )

    err_radial = np.linalg.norm(est_radial.vector - v_true)
    err_pos = np.linalg.norm(est_pos.vector - v_true)
    err_fused = np.linalg.norm(est_fused.vector - v_true)

    # Fused should be at least as good as the better single-source estimate
    # for this noise profile.
    assert err_fused <= min(err_radial, err_pos) + 1e-6


def test_xyvr_high_level_interface() -> None:
    v_true = np.array([1.6, -0.4])
    positions = np.array([[8.0, 1.0], [5.0, 6.0], [-2.0, 8.0], [-7.0, 3.0]])
    los = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    vr = los @ v_true
    points = np.column_stack([positions, vr])

    est = estimate_velocity_from_xyvr(points)

    np.testing.assert_allclose(est.vector, v_true, atol=1e-12)


def test_txyvr_high_level_fused_interface() -> None:
    rng = np.random.default_rng(99)
    t = np.linspace(0.0, 3.0, 20)
    p0 = np.array([10.0, -3.0])
    v_true = np.array([2.1, 0.7])
    p_true = p0 + t[:, None] * v_true
    los = p_true / np.linalg.norm(p_true, axis=1, keepdims=True)
    vr_true = los @ v_true

    p_meas = p_true + rng.normal(0.0, 0.1, size=p_true.shape)
    vr_meas = vr_true + rng.normal(0.0, 0.05, size=vr_true.shape)
    samples = np.column_stack([t, p_meas, vr_meas])

    est = estimate_velocity_from_time_xyvr(
        samples,
        weight_position=1.0 / (0.1**2),
        weight_radial=1.0 / (0.05**2),
    )

    assert np.linalg.norm(est.vector - v_true) < 0.2


def test_non_finite_values_raise() -> None:
    with pytest.raises(EstimationError, match="finite"):
        estimate_velocity_from_xyvr(np.array([[1.0, 2.0, np.nan], [2.0, 3.0, 0.1]]))

    with pytest.raises(EstimationError, match="finite"):
        estimate_velocity_from_time_xyvr(
            np.array([[0.0, 1.0, 2.0, 0.1], [1.0, np.inf, 3.0, 0.2]])
        )


def test_zero_range_position_raises() -> None:
    # Point at the radar origin has undefined line-of-sight direction.
    points = np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.2]])
    with pytest.raises(EstimationError, match="non-zero distance"):
        estimate_velocity_from_xyvr(points)
