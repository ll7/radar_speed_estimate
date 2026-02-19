## **Executive Summary**

Yes—under your assumptions (stationary sensor, noise-free, constant-velocity target, and **at least two time-stamped 2D position measurements**), the target’s **true velocity vector** (orientation and magnitude in the sensor frame) is uniquely determined. In fact, the **positions alone** are sufficient; the radial (toward-us) velocity is not required for identifiability, but it provides a consistency check (and helps under noise).

If, instead, you had **only radial velocity** (range-rate) at one time, the tangential component is unobservable. With multiple radial-velocity measurements at different aspect angles, you can recover the full velocity provided the lines of sight are sufficiently diverse.

## **1) Using measured positions (x, y) at multiple times**

Let the measured target position in the radar frame be p_i = [x_i, y_i]^T at time t_i. Under constant velocity v = [v_x, v_y]^T,

p(t) = p(t_0) + v (t - t_0).

With **two** measurements (p_1, t_1), (p_2, t_2) and \Delta t = t_2 - t_1 \neq 0,

v = \frac{p_2 - p_1}{\Delta t}.

Then

- speed: \|v\| = \sqrt{v_x^2 + v_y^2}
- orientation (heading in radar frame): \psi = \mathrm{atan2}(v_y, v_x)

With **more than two** measurements, you can fit v via least squares (still exact if noise-free and perfectly constant-velocity).

### **Observability comment**

With 2D positions at known times and constant velocity, the velocity is observable from two distinct time samples (unless \Delta t = 0).

## **2) Using only radial velocity + positions (no differencing needed)**

If you want to use the “radial component” relation explicitly:

- Line-of-sight unit vector at measurement i:
  u_i = \frac{p_i}{\|p_i\|}.
- Radar-measured radial velocity (toward/away) is the projection:
  v\_{r,i} = u_i^T v.

Each measurement gives a linear constraint on v:

u*i^T v = v*{r,i}.

Stacking N measurements:

U v = v*r, \quad U = \begin{bmatrix} u_1^T\\ \vdots\\ u_N^T \end{bmatrix}, \quad v_r= \begin{bmatrix} v*{r,1}\\ \vdots\\ v\_{r,N} \end{bmatrix}.

- In **2D**, you need at least **two** measurements with **non-collinear** u_i (i.e., changing aspect angle) for a unique solution.
- Solve (noise-free, exactly determined) with two equations, or (overdetermined) via least squares:
  v = (U^T U)^{-1} U^T v_r
  provided U^T U is invertible (rank 2 in 2D).

### **Degenerate case (unobservable tangential component)**

If all u_i are (nearly) the same direction (target stays on the same ray from the radar), then \mathrm{rank}(U)=1 and you can only recover the radial component, not the tangential component, **from radial-velocity constraints alone**.

## **3) Consistency check between the two information sources**

Given v from position differencing, the predicted radial velocity is

\hat v\_{r,i} = u_i^T v,

which should match the measured v\_{r,i} in the noise-free idealization. Discrepancies indicate one of:

- non-constant velocity (acceleration/turning),
- time alignment issues,
- coordinate-frame issues (e.g., sensor not truly stationary, wrong ego compensation),
- association errors (not the same object).

## **Practical takeaways**

- **If you truly have** x,y **positions at** \ge 2 **times**: compute v directly from \Delta p/\Delta t. This yields full magnitude + orientation.
- **If you had only** v_r **at one time**: cannot recover tangential component.
- **If you have** v_r **at multiple times** and aspect changes: you can recover v (2D needs two independent LOS directions; 3D needs three).

## **Optional follow-up tasks**

- Provide a compact estimator for constant-velocity v and p_0 from N samples (closed-form least squares).
- Extend to constant-acceleration (CA) motion and derive minimal measurement counts / observability conditions.
- Discuss what changes if the sensor is moving (ego motion compensation, relative vs absolute velocity).
