from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radar_speed_estimate import estimate_velocity_from_xyvr


def rotation_matrix(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def main() -> None:
    """Rigid-object multi-point radar example.

    Scenario:
    - Radar at origin.
    - A rigid rectangular vehicle is located away from radar.
    - Four corner points are measured simultaneously (not collinear).
    - All points share the same translational velocity.

    For each point i:
        u_i = p_i / ||p_i||
        v_r,i = u_i^T v            (positive = away from radar)
        v_towards,i = -v_r,i       (positive = towards radar)
    """

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rigid object geometry in local frame (rectangle corners).
    length_m = 4.5
    width_m = 2.0
    local_corners = np.array(
        [
            [+length_m / 2, +width_m / 2],
            [+length_m / 2, -width_m / 2],
            [-length_m / 2, -width_m / 2],
            [-length_m / 2, +width_m / 2],
        ]
    )

    # Pose in world/radar frame.
    center = np.array([18.0, 24.0])
    yaw_deg = 32.0
    R = rotation_matrix(np.deg2rad(yaw_deg))
    corners = (local_corners @ R.T) + center

    # Shared rigid-body translational velocity.
    speed_mps = 10.0
    heading_deg = 215.0
    heading_rad = np.deg2rad(heading_deg)
    v_true = speed_mps * np.array([np.cos(heading_rad), np.sin(heading_rad)])

    # Exclude p0 from measurements to mimic a typical visibility/reflectivity case.
    # We still draw the full rigid body, but only points p1..p3 are used as returns.
    measured_idx = np.array([1, 2, 3])
    measured_points = corners[measured_idx]
    los = measured_points / np.linalg.norm(measured_points, axis=1, keepdims=True)
    v_radial_away = los @ v_true
    v_towards = -v_radial_away

    points_xyvr = np.column_stack([measured_points, v_radial_away])
    estimate = estimate_velocity_from_xyvr(points_xyvr)

    # Save scenario table.
    csv_path = out_dir / "scenario_rigid_body_points_xyvr.csv"
    header = "point_idx,x_m,y_m,v_r_away_mps,v_towards_mps,los_x,los_y"
    rows = np.column_stack(
        [
            measured_idx,
            measured_points,
            v_radial_away,
            v_towards,
            los,
        ]
    )
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")

    # Figure with geometry and towards-radar projection vectors.
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(0.0, 0.0, marker="*", s=280, color="black", label="radar")

    # Draw rigid body polygon.
    poly = np.vstack([corners, corners[0]])
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=2, label="rigid object boundary")
    ax.scatter(
        measured_points[:, 0],
        measured_points[:, 1],
        color="#1f77b4",
        zorder=3,
        label="measured points",
    )

    # LOS rays and projection vectors.
    for local_i, p in enumerate(measured_points):
        i = int(measured_idx[local_i])
        ax.plot([0.0, p[0]], [0.0, p[1]], color="#4c8bf5", alpha=0.55, lw=2)

        # Vector projection along direction towards radar: s*(-u), where s=v_towards.
        proj_vec = v_towards[local_i] * (-los[local_i])
        ax.quiver(
            p[0],
            p[1],
            proj_vec[0],
            proj_vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#d62728",
            width=0.005,
            alpha=0.85,
        )
        ax.text(
            p[0] + 0.25,
            p[1] + 0.2,
            f"p{i}: v_tow={v_towards[local_i]:+.2f}",
            fontsize=9,
            color="#d62728",
        )

    # True and estimated global velocity vectors anchored at center.
    ax.quiver(
        center[0],
        center[1],
        v_true[0],
        v_true[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#2ca02c",
        width=0.006,
        label="true object velocity",
    )
    ax.quiver(
        center[0],
        center[1],
        estimate.vx,
        estimate.vy,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#ff7f0e",
        width=0.006,
        label="estimated velocity",
    )

    ax.set_title(
        "Rigid-object multi-point radar scenario\n"
        "Blue: LOS rays, Red: projection towards radar along each LOS"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Right panel: scalar projections summary.
    ax2 = fig.add_subplot(gs[0, 1])
    idx = measured_idx
    ax2.bar(idx - 0.18, v_radial_away, width=0.36, label="v_r (away +)", color="#1f77b4")
    ax2.bar(idx + 0.18, v_towards, width=0.36, label="v_towards (+)", color="#d62728")
    ax2.axhline(0.0, color="black", lw=1)
    ax2.set_xticks(idx)
    ax2.set_xlabel("point index")
    ax2.set_ylabel("velocity component [m/s]")
    ax2.set_title("Per-point LOS projections")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(fontsize=9)

    fig_path = out_dir / "scenario_rigid_body_points.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    summary = out_dir / "scenario_rigid_body_points_summary.txt"
    summary.write_text(
        "\n".join(
            [
                "Rigid-body point scenario",
                f"center={center.tolist()}, yaw_deg={yaw_deg}",
                f"speed_mps={speed_mps}, heading_deg={heading_deg}",
                f"v_true={v_true.tolist()}",
                f"v_estimated={estimate.vector.tolist()}",
                f"radial_away={v_radial_away.tolist()}",
                f"radial_towards={v_towards.tolist()}",
            ]
        )
    )

    print(f"saved plot: {fig_path}")
    print(f"saved csv: {csv_path}")
    print(f"saved summary: {summary}")


if __name__ == "__main__":
    main()
