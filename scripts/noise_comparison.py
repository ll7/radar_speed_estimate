from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radar_speed_estimate import (
    estimate_velocity_fused,
    estimate_velocity_from_positions,
    estimate_velocity_from_radial,
)


def run_trial(rng: np.random.Generator, n: int, pos_sigma: float, vr_sigma: float) -> tuple[float, float, float]:
    t = np.linspace(0.0, 5.0, n)
    p0 = np.array([18.0, -8.0])
    v_true = np.array([-3.0, 1.4])

    p_true = p0 + t[:, None] * v_true
    los = p_true / np.linalg.norm(p_true, axis=1, keepdims=True)
    vr_true = los @ v_true

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

    err_r = float(np.linalg.norm(est_r.vector - v_true))
    err_p = float(np.linalg.norm(est_p.vector - v_true))
    err_f = float(np.linalg.norm(est_f.vector - v_true))
    return err_r, err_p, err_f


def main() -> None:
    rng = np.random.default_rng(123)
    n_list = [2, 3, 5, 10, 20, 40, 80]
    trials = 250
    pos_sigma = 0.35
    vr_sigma = 0.12

    radial_mean = []
    pos_mean = []
    fused_mean = []

    for n in n_list:
        errs_r = []
        errs_p = []
        errs_f = []
        for _ in range(trials):
            er, ep, ef = run_trial(rng, n=n, pos_sigma=pos_sigma, vr_sigma=vr_sigma)
            errs_r.append(er)
            errs_p.append(ep)
            errs_f.append(ef)
        radial_mean.append(float(np.mean(errs_r)))
        pos_mean.append(float(np.mean(errs_p)))
        fused_mean.append(float(np.mean(errs_f)))

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(n_list, radial_mean, "o-", label="radial only")
    plt.plot(n_list, pos_mean, "o-", label="position-time only")
    plt.plot(n_list, fused_mean, "o-", label="fused (second approach)")
    plt.xlabel("number of measurements")
    plt.ylabel("mean velocity vector error [m/s]")
    plt.title("Noisy estimation comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "noise_comparison.png"
    plt.savefig(fig_path, dpi=180)

    report_path = out_dir / "noise_comparison.txt"
    lines = [
        "Noisy velocity estimation comparison",
        f"trials={trials}, pos_sigma={pos_sigma}, vr_sigma={vr_sigma}",
        "n,radial_mean_error,position_mean_error,fused_mean_error",
    ]
    for n, er, ep, ef in zip(n_list, radial_mean, pos_mean, fused_mean):
        lines.append(f"{n},{er:.6f},{ep:.6f},{ef:.6f}")
    report_path.write_text("\n".join(lines))

    print(f"saved plot: {fig_path}")
    print(f"saved summary: {report_path}")


if __name__ == "__main__":
    main()
