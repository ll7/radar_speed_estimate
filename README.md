# radar_speed_estimate

Radar true-velocity estimation utilities (ROS2 Jazzy-compatible Python 3.12) using:

- radial Doppler + LOS geometry (`U v = v_r`)
- position/time constant-velocity fit
- fused weighted least-squares (position + radial, noise-aware second approach)

## Data convention

- Coordinates: `x, y` in meters in radar frame.
- Time: `t` in seconds.
- Radial velocity: `v_r` in m/s.
- Sign convention: **positive `v_r` means motion away from the radar origin** (because `v_r = u^T v` with `u` pointing from radar to target).

## Context sources

This repo integrates the following concept notes:

- `context/vault_notes/noise free radar speed estimate perplexity.md`
- `context/vault_notes/perplexity radar true velocity 2026-02-19.md`
- `context/vault_notes/gemini radar true velocity 2026-02-19.md`
- `context/vault_notes/chatgpt radar true velocity 2026-02-19.md`

Consolidated note intake:

- `context/2026-02-19_obsidian_radar_velocity_notes.md`

## Environment setup (uv)

```bash
uv venv --python 3.12
uv sync
```

Python version is pinned to 3.12 via `.python-version`.

## Run tests

```bash
uv run pytest -q
```

## Run visualization

```bash
uv run python scripts/visualize_estimation.py
uv run python scripts/noise_comparison.py
uv run python scripts/scenario_rigid_body_points.py
```

Generated artifacts:

- `artifacts/tutorial_velocity_estimation_overview.png`
- `artifacts/tutorial_velocity_estimation_overview.txt`
- `artifacts/noise_comparison.png`
- `artifacts/noise_comparison.txt`
- `artifacts/scenario_rigid_body_points.png`
- `artifacts/scenario_rigid_body_points_xyvr.csv`
- `artifacts/scenario_rigid_body_points_summary.txt`

## Main API

- `estimate_velocity_from_radial(...)`
- `estimate_velocity_from_positions(...)`
- `estimate_velocity_fused(...)`
- `estimate_velocity_from_xyvr(...)` (input: `N x 3` -> columns `[x, y, v_r]`)
- `estimate_velocity_from_time_xyvr(...)` (input: `N x 4` -> `[t, x, y, v_r]`)
- `radial_consistency_error(...)`

### Quick interface example (`N x 3`)

```python
import numpy as np
from radar_speed_estimate import estimate_velocity_from_xyvr

# Create a self-consistent synthetic example from one true velocity.
v_true = np.array([2.0, -1.0])  # [vx, vy] in m/s
positions = np.array([
    [10.0, 0.0],
    [7.0, 7.0],
    [0.0, 9.0],
])
los = positions / np.linalg.norm(positions, axis=1, keepdims=True)
radial = los @ v_true

points_xyvr = np.array([
    [positions[0, 0], positions[0, 1], radial[0]],
    [positions[1, 0], positions[1, 1], radial[1]],
    [positions[2, 0], positions[2, 1], radial[2]],
])

estimate = estimate_velocity_from_xyvr(points_xyvr)
print(estimate.vx, estimate.vy, estimate.speed, estimate.heading_rad)
```

## Documentation

- Student tutorial: `docs/tutorial.md`
- Method details: `docs/method.md`
- Testing guide: `docs/testing.md`
- Validation record: `docs/validation_report.md`
