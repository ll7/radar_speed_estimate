from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radar_speed_estimate import (
    estimate_velocity_fused,
    estimate_velocity_from_positions,
    estimate_velocity_from_radial,
    radial_consistency_error,
)


def _simulate_clean_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic, noise-free constant-velocity scenario."""
    t = np.array([0.0, 0.7, 1.4, 2.2, 3.1, 4.0])
    p0 = np.array([16.0, -7.0])
    v_true = np.array([-2.5, 1.1])
    positions = p0 + t[:, None] * v_true
    return t, positions, v_true


def _noisy_errors_by_n(rng: np.random.Generator) -> tuple[list[int], list[float], list[float], list[float]]:
    """Monte-Carlo mean errors vs number of measurements for three estimators."""
    n_list = [2, 3, 5, 10, 20, 40]
    trials = 200
    pos_sigma = 0.35
    vr_sigma = 0.12

    radial_mean: list[float] = []
    pos_mean: list[float] = []
    fused_mean: list[float] = []

    for n in n_list:
        errs_r = []
        errs_p = []
        errs_f = []
        t = np.linspace(0.0, 5.0, n)
        p0 = np.array([18.0, -8.0])
        v_true = np.array([-3.0, 1.4])
        p_true = p0 + t[:, None] * v_true
        los_true = p_true / np.linalg.norm(p_true, axis=1, keepdims=True)
        vr_true = los_true @ v_true

        for _ in range(trials):
            p_meas = p_true + rng.normal(0.0, pos_sigma, size=p_true.shape)
            vr_meas = vr_true + rng.normal(0.0, vr_sigma, size=vr_true.shape)

            est_r = estimate_velocity_from_radial(p_meas, vr_meas)
            est_p = estimate_velocity_from_positions(t, p_meas)
            est_f = estimate_velocity_fused(
                t,
                p_meas,
                vr_meas,
                weight_position=1.0 / (pos_sigma**2),
                weight_radial=1.0 / (vr_sigma**2),
            )

            errs_r.append(float(np.linalg.norm(est_r.vector - v_true)))
            errs_p.append(float(np.linalg.norm(est_p.vector - v_true)))
            errs_f.append(float(np.linalg.norm(est_f.vector - v_true)))

        radial_mean.append(float(np.mean(errs_r)))
        pos_mean.append(float(np.mean(errs_p)))
        fused_mean.append(float(np.mean(errs_f)))

    return n_list, radial_mean, pos_mean, fused_mean


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    t, positions, v_true = _simulate_clean_case()
    los = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    vr = los @ v_true

    est_radial = estimate_velocity_from_radial(positions, vr)
    est_pos = estimate_velocity_from_positions(t, positions)
    residuals = radial_consistency_error(est_radial.vector, positions, vr)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.27, wspace=0.22)

    # Panel A: 2D geometry, LOS rays and velocity vectors.
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(0.0, 0.0, marker="*", s=220, color="black", label="radar origin")
    ax_a.plot(positions[:, 0], positions[:, 1], "o-", color="#1f77b4", label="target track")
    for i in range(len(positions)):
        ax_a.plot([0.0, positions[i, 0]], [0.0, positions[i, 1]], "-", alpha=0.2, color="gray")
        ax_a.text(positions[i, 0] + 0.25, positions[i, 1] + 0.1, f"k={i}", fontsize=8)

    anchor = positions[0]
    ax_a.quiver(
        anchor[0],
        anchor[1],
        v_true[0],
        v_true[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        color="#2ca02c",
        label="true v",
    )
    ax_a.quiver(
        anchor[0],
        anchor[1],
        est_radial.vx,
        est_radial.vy,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        color="#d62728",
        label="estimated v (radial)",
    )
    ax_a.set_title("A) Geometry: changing LOS makes 2D velocity observable")
    ax_a.set_xlabel("x [m]")
    ax_a.set_ylabel("y [m]")
    ax_a.axis("equal")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="best", fontsize=9)

    # Panel B: radial projection equation intuition.
    ax_b = fig.add_subplot(gs[0, 1])
    indices = np.arange(len(vr))
    ax_b.bar(indices - 0.18, vr, width=0.35, label=r"measured $v_{r,k}$", color="#1f77b4")
    proj_true = los @ v_true
    ax_b.bar(indices + 0.18, proj_true, width=0.35, label=r"$u_k^\top v_{true}$", color="#ff7f0e")
    ax_b.set_title(r"B) Each sample adds one equation: $v_{r,k}=u_k^\top v$")
    ax_b.set_xlabel("sample index k")
    ax_b.set_ylabel("radial velocity [m/s]")
    ax_b.grid(True, axis="y", alpha=0.25)
    ax_b.legend(fontsize=9)

    # Panel C: time-domain radial consistency.
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(t, vr, "o-", color="#1f77b4", label="measured radial velocity")
    ax_c.plot(t, vr + residuals, "x--", color="#d62728", label="predicted from estimated v")
    ax_c.set_title("C) Consistency check: predicted vs measured Doppler")
    ax_c.set_xlabel("time [s]")
    ax_c.set_ylabel("radial velocity [m/s]")
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(fontsize=9)

    # Panel D: noisy-data comparison by sample count.
    rng = np.random.default_rng(123)
    n_list, radial_mean, pos_mean, fused_mean = _noisy_errors_by_n(rng)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(n_list, radial_mean, "o-", label="radial only", color="#d62728")
    ax_d.plot(n_list, pos_mean, "o-", label="position-time only", color="#1f77b4")
    ax_d.plot(n_list, fused_mean, "o-", label="fused", color="#2ca02c")
    ax_d.set_title("D) Under noise: more samples help, fusion is strongest")
    ax_d.set_xlabel("number of measurements N")
    ax_d.set_ylabel("mean velocity error [m/s]")
    ax_d.grid(True, alpha=0.25)
    ax_d.legend(fontsize=9)

    fig.suptitle(
        "Radar True-Velocity Estimation Tutorial Figure\n"
        + rf"$v_{{true}}=[{v_true[0]:.2f},{v_true[1]:.2f}]$, "
        + rf"$\hat v_{{radial}}=[{est_radial.vx:.2f},{est_radial.vy:.2f}]$, "
        + rf"$\hat v_{{pos}}=[{est_pos.vx:.2f},{est_pos.vy:.2f}]$",
        fontsize=13,
    )

    fig_path = out_dir / "tutorial_velocity_estimation_overview.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    summary = out_dir / "tutorial_velocity_estimation_overview.txt"
    summary.write_text(
        "\n".join(
            [
                "Tutorial velocity estimation overview",
                f"true_velocity={v_true.tolist()}",
                f"estimated_from_radial={est_radial.vector.tolist()}",
                f"estimated_from_positions={est_pos.vector.tolist()}",
                f"radial_l2_residual={est_radial.residual_l2:.6e}",
                f"condition_number_U={est_radial.condition_number:.6e}",
                "panel_D: mean errors for N=[2,3,5,10,20,40]",
                f"radial_mean_errors={radial_mean}",
                f"position_mean_errors={pos_mean}",
                f"fused_mean_errors={fused_mean}",
            ]
        )
    )

    print(f"saved plot: {fig_path}")
    print(f"saved summary: {summary}")


if __name__ == "__main__":
    main()
