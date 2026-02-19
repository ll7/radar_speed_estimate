from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radar_speed_estimate import estimate_velocity_from_xyvr


def main() -> None:
    """Create a deterministic radar scenario from user-specified geometry.

    Assumptions used:
    - Radar at origin.
    - Target speed = 10 m/s.
    - Target velocity heading = 45 deg in radar frame.
    - Initial target distance from radar = 20 m.
    - Initial bearing (radar -> target) = 140 deg.
    """

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    speed_mps = 10.0
    velocity_heading_deg = 45.0
    initial_range_m = 20.0
    initial_bearing_deg = 140.0

    # Build true velocity vector.
    v_heading_rad = np.deg2rad(velocity_heading_deg)
    v_true = speed_mps * np.array([np.cos(v_heading_rad), np.sin(v_heading_rad)])

    # Initial target position from polar coordinates.
    b_rad = np.deg2rad(initial_bearing_deg)
    p0 = initial_range_m * np.array([np.cos(b_rad), np.sin(b_rad)])

    # Generate samples over time for constant-velocity target.
    t = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    positions = p0 + t[:, None] * v_true

    # Radial measurements from projection model v_r = u^T v.
    los = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    radial_velocity = los @ v_true

    # N x 3 interface: [x, y, v_r]
    points_xyvr = np.column_stack([positions, radial_velocity])
    estimate = estimate_velocity_from_xyvr(points_xyvr)

    # Angle between initial LOS and velocity vector (for interpretation).
    los0 = los[0]
    cos_alpha = float(np.clip(np.dot(los0, v_true) / np.linalg.norm(v_true), -1.0, 1.0))
    alpha_deg = float(np.rad2deg(np.arccos(cos_alpha)))

    # Save tabular data.
    csv_path = out_dir / "scenario_45deg_140deg_points_xyvr.csv"
    np.savetxt(
        csv_path,
        np.column_stack([t, points_xyvr]),
        delimiter=",",
        header="t_s,x_m,y_m,v_r_mps",
        comments="",
    )

    # Save summary.
    summary_path = out_dir / "scenario_45deg_140deg_summary.txt"
    summary_lines = [
        "Scenario: heading 45 deg, speed 10 m/s, range 20 m, bearing 140 deg",
        f"v_true={v_true.tolist()}",
        f"v_estimated={estimate.vector.tolist()}",
        f"initial_position={p0.tolist()}",
        f"initial_los_vs_velocity_angle_deg={alpha_deg:.6f}",
        f"radial_samples={radial_velocity.tolist()}",
    ]
    summary_path.write_text("\n".join(summary_lines))

    # Visualization.
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.scatter(0.0, 0.0, marker="*", s=220, color="black", label="radar")
    ax.plot(positions[:, 0], positions[:, 1], "o-", color="#1f77b4", label="target path")

    for i in range(len(positions)):
        ax.plot([0.0, positions[i, 0]], [0.0, positions[i, 1]], color="gray", alpha=0.25)
        ax.text(positions[i, 0] + 0.2, positions[i, 1] + 0.2, f"k={i}", fontsize=8)

    ax.quiver(
        positions[0, 0],
        positions[0, 1],
        v_true[0],
        v_true[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        color="#2ca02c",
        label="true velocity",
    )
    ax.quiver(
        positions[0, 0],
        positions[0, 1],
        estimate.vx,
        estimate.vy,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        color="#d62728",
        label="estimated velocity",
    )

    ax.set_title(
        "Scenario: v=10 m/s @ 45 deg, initial range=20 m @ bearing 140 deg\n"
        + f"angle(initial LOS, velocity)={alpha_deg:.1f} deg"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig_path = out_dir / "scenario_45deg_140deg.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")

    print(f"saved plot: {fig_path}")
    print(f"saved csv: {csv_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
