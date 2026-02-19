# Method

## Problem setup

Given target positions $p_i=(x_i,y_i)$ from a stationary radar and measured radial velocities $v_{r,i}$, estimate the true 2D velocity vector $v=(v_x,v_y)$.

## Radial-LOS estimator

For each sample `i`, form line-of-sight unit vector:

$$
u_i = \frac{p_i}{\|p_i\|}.
$$

Measurement model:

$$
v_{r,i} = u_i^\top v.
$$

Stacking all samples:

$$
U v = v_r.
$$

Solve with least squares:

$$
\hat v = \arg\min_v \|U v - v_r\|_2.
$$

Observability/quality gates:

- require $\mathrm{rank}(U)=2$ (full observability in 2D)
- reject high condition number ($\mathrm{cond}(U) > 10^6$)

## Position-time estimator

Under constant velocity:

$$
p(t) = p_0 + v t.
$$

Estimate `v_x` and `v_y` by linear least squares on x(t), y(t).
Estimate $v_x$ and $v_y$ by linear least squares on $x(t), y(t)$.

## Fused estimator (second approach for noisy data)

Joint weighted least squares with unknowns `[x0, y0, vx, vy]`:
Joint weighted least squares with unknowns $[x_0, y_0, v_x, v_y]$:

- Position equations:
  - $x_i = x_0 + v_x t_i$
  - $y_i = y_0 + v_y t_i$
- Radial equations:
  - $u_i^\top [v_x, v_y]^\top = v_{r,i}$

This fuses both information sources and is typically more stable under noise when weights reflect sensor variance.

## Diagnostics

The implementation reports:

- residual norm
- matrix rank
- condition number
- speed and heading angle (`atan2(v_y, v_x)`)
