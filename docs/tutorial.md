# Student Tutorial: Estimating True 2D Velocity from Radar Measurements

This tutorial explains the concept step by step and then walks you through running it in this repository.

## Learning goals

By the end, you should be able to:

1. Explain why a single radial velocity measurement is not enough for full 2D velocity.
2. Reconstruct true velocity from two or more measurements.
3. Compare three estimators:
   - radial-only (`U v = v_r`)
   - position-time (constant velocity fit)
   - fused weighted least squares (better under noise)
4. Run tests and generate plots.

## Measurement convention used in this tutorial

- Position: $(x, y)$ in meters in radar frame.
- Time: $t$ in seconds.
- Radial velocity: $v_r$ in m/s.
- Sign: $v_r > 0$ means the target moves **away** from radar origin.

## 1. Intuition first

Radar Doppler gives velocity **toward/away** from the radar (radial component), not sideways motion.

- One measurement gives one projection equation, so 2D velocity is underdetermined.
- If you take more measurements at different line-of-sight directions, each gives a new equation.
- With enough geometric diversity, you can solve for `v_x, v_y`.

## 2. Mathematical model

For measurement `i`:

- position: $p_i = (x_i, y_i)$
- line-of-sight unit vector: $u_i = \frac{p_i}{\|p_i\|}$
- measured radial velocity: $v_{r,i}$

Model:

$$
v_{r,i} = u_i^\top v,\quad \text{where } v = \begin{bmatrix} v_x \\ v_y \end{bmatrix}.
$$

Stack all measurements:

$$
U v = v_r.
$$

Solve with least squares:

$$
\hat v = \arg\min_v \|U v - v_r\|_2.
$$

Important condition in 2D:

- Need at least 2 samples with non-collinear LOS vectors.
- If all LOS directions are the same ($\mathrm{rank}(U)=1$), tangential velocity is unobservable.

## 3. Repository setup (uv + Python 3.12)

From repo root:

```bash
uv venv --python 3.12
uv sync
```

Check Python:

```bash
uv run python -V
```

Expected: Python 3.12.x.

## 4. Step-by-step code tour

Main implementation:

- `src/radar_speed_estimate/estimator.py`

Functions:

1. `estimate_velocity_from_radial(...)`
   - Uses $U v = v_r$.
   - Checks rank/conditioning.
2. `estimate_velocity_from_positions(...)`
   - Fits
     $$
     x(t)=x_0+v_x t,\qquad y(t)=y_0+v_y t.
     $$
3. `estimate_velocity_fused(...)`
   - Joint weighted least squares using both position and radial equations.
4. `radial_consistency_error(...)`
   - Residual check for predicted vs measured radial velocity.

## 5. Run unit tests

```bash
uv run pytest -q
```

What is validated:

- exact reconstruction from 2 measurements
- exact reconstruction from overdetermined systems
- degeneracy detection ($\mathrm{rank}(U) < 2$)
- position-based constant-velocity fit
- randomized Monte Carlo stability
- noisy case with fused estimator comparison

## 6. Visual demo (noise-free concept)

```bash
uv run python scripts/visualize_estimation.py
```

Outputs:

- `artifacts/tutorial_velocity_estimation_overview.png`
- `artifacts/tutorial_velocity_estimation_overview.txt`

How to read the tutorial figure:

- **Panel A (Geometry):** Radar is at origin, gray rays are LOS directions, blue points are target samples.  
  Key takeaway: changing LOS direction across samples makes full 2D velocity observable.
- **Panel B (Equation intuition):** Bars show measured Doppler values and projection values $u_k^\top v$.  
  Key takeaway: each sample contributes one linear equation in $(v_x, v_y)$.
- **Panel C (Consistency):** Predicted radial velocities from estimated $v$ are compared with measured Doppler over time.  
  Key takeaway: near-overlap means estimator and measurements agree.
- **Panel D (Noise behavior):** Mean velocity error vs number of samples for radial-only, position-only, and fused methods.  
  Key takeaway: more samples improve robustness; fused estimation is best under this noise profile.

## 7. Noisy multi-measurement experiment

```bash
uv run python scripts/noise_comparison.py
```

Outputs:

- `artifacts/noise_comparison.png`
- `artifacts/noise_comparison.txt`

Interpretation:

- As sample count increases, all methods improve.
- Fused method is usually best (or close to best) because it combines both data sources.

## 8. Rigid-object multi-point example (non-collinear points)

```bash
uv run python scripts/scenario_rigid_body_points.py
```

Outputs:

- `artifacts/scenario_rigid_body_points.png`
- `artifacts/scenario_rigid_body_points_xyvr.csv`
- `artifacts/scenario_rigid_body_points_summary.txt`

What to observe:

- Measurement points are object corners (same rigid body), not collinear.
- Point `p0` is intentionally excluded from radar returns in this example.
- All points share one translational velocity vector.
- Each LOS line has its own projected component toward radar:
  $$
  v_{\text{towards},i} = -\,u_i^\top v.
  $$
- The plot overlays these projection vectors along each LOS and compares true vs estimated object velocity.

## 9. Suggested student exercises

1. Change noise levels in `scripts/noise_comparison.py` and rerun.
2. Reduce measurement count to 2 and observe stability.
3. Create nearly collinear LOS geometry and confirm degeneracy behavior.
4. Tune fused weights and compare error curves.
5. Add synthetic acceleration (non-constant velocity) and inspect residual growth.

## 10. Concept summary

1. Two or more measurements are enough in 2D only if LOS geometry is informative.
2. More measurements improve robustness.
3. Under noise, fused estimation is a practical second approach.
4. Diagnostics (rank, condition number, residuals) are essential to trust estimates.
