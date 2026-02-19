# Context Intake: Radar True Velocity Notes (Obsidian)

Date: 2026-02-19

Source notes (MyVaultLL2):

- `context/vault_notes/noise free radar speed estimate perplexity.md`
- `context/vault_notes/perplexity radar true velocity 2026-02-19.md`
- `context/vault_notes/gemini radar true velocity 2026-02-19.md`
- `context/vault_notes/chatgpt radar true velocity 2026-02-19.md`

## Consolidated technical conclusions

1. **Radial projection model**:
   For each measurement i, `v_r,i = u_i^T v`, where `u_i` is line-of-sight unit vector and `v=[v_x,v_y]^T`.
2. **Identifiability condition**:
   In 2D, at least two measurements with non-collinear LOS vectors are required for unique recovery from radial data.
3. **Least-squares formulation**:
   Stack rows `u_i^T` into `U`, then solve `U v = v_r`; overdetermined cases use least squares.
4. **Degenerate case**:
   If all LOS vectors are parallel (`rank(U)=1`), tangential component is unobservable.
5. **Position/time alternative**:
   Under constant velocity and known timestamps, `v` is directly observable from position trajectory (`Δp/Δt` or linear fit).
6. **Consistency check**:
   Given an estimated `v`, compare predicted radial values `u_i^T v` with measured `v_r,i`.

## Implementation policy in this repository

- Implement both estimators:
  - radial+LOS least-squares (`U v = v_r`)
  - position/time least-squares under constant velocity
- Reject degenerate/ill-conditioned LOS geometry.
- Provide diagnostics (rank, condition number, residuals).
- Validate with deterministic and randomized tests.
- Provide a visualization artifact for quick inspection.
