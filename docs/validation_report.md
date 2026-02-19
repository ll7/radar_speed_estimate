# Validation Report

Date: 2026-02-19

## Environment

- Tooling: `uv`
- Virtual environment: `.venv` (repository-local)
- Python: `3.12.9` (ROS2 Jazzy-compatible family: Python 3.12)

## Commands executed

```bash
uv run python -V
uv run pytest -q
uv run python scripts/visualize_estimation.py
uv run python scripts/noise_comparison.py
uv run python scripts/scenario_rigid_body_points.py
```

## Results

- Test status: `11 passed`
- Generated tutorial plot: `artifacts/tutorial_velocity_estimation_overview.png`
- Generated tutorial summary: `artifacts/tutorial_velocity_estimation_overview.txt`
- Generated noisy-comparison plot: `artifacts/noise_comparison.png`
- Generated noisy-comparison summary: `artifacts/noise_comparison.txt`
- Generated rigid-body plot: `artifacts/scenario_rigid_body_points.png`
- Generated rigid-body data: `artifacts/scenario_rigid_body_points_xyvr.csv`
- Generated rigid-body summary: `artifacts/scenario_rigid_body_points_summary.txt`

## Implemented estimators

- `estimate_velocity_from_radial(...)`
  - Uses LOS unit vectors and solves `U v = v_r` via least squares.
  - Rejects degenerate geometry (`rank(U) < 2`).
  - Rejects ill-conditioned geometry (`cond(U) > 1e6`).
- `estimate_velocity_from_positions(...)`
  - Fits constant-velocity model from `(t, x, y)` via least squares.
- `estimate_velocity_fused(...)`
  - Joint weighted least-squares over position and radial constraints.
- `estimate_velocity_from_xyvr(...)`
  - High-level interface for `N x 3` points (`[x, y, v_r]`).
- `estimate_velocity_from_time_xyvr(...)`
  - High-level fused interface for `N x 4` samples (`[t, x, y, v_r]`).
- `radial_consistency_error(...)`
  - Compares predicted and measured radial velocities for diagnostics.
