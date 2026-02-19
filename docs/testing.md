# Testing and Validation

Run all tests:

```bash
uv run pytest -q
```

Run visualization demo:

```bash
uv run python scripts/visualize_estimation.py
uv run python scripts/noise_comparison.py
```

Expected outputs:

- `artifacts/tutorial_velocity_estimation_overview.png`
- `artifacts/tutorial_velocity_estimation_overview.txt`
- `artifacts/noise_comparison.png`
- `artifacts/noise_comparison.txt`

## Test coverage highlights

- exact 2-sample radial reconstruction
- overdetermined radial least-squares reconstruction
- degeneracy handling for collinear LOS
- exact position/time constant-velocity fit
- radial consistency check
- randomized Monte Carlo stability sweep (100 trials)
- noisy many-measurement test with fused estimator
