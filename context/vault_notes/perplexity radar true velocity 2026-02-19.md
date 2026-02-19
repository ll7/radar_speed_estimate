Yes, but only if those multiple measurements give you **different viewing angles** to the same constant‑velocity target (i.e., the object has moved so that the line of sight changes). Otherwise the problem is under‑determined.

---

## Geometry and equations

Let the target have constant (unknown) velocity vector v=(vx,vy)v = (v*x, v_y)v=(vx,vy).  
At time tkt_ktk you know its position pk=(xk,yk)p_k = (x_k, y_k)pk=(xk,yk) and the measured radial velocity vr,kv*{r,k}vr,k (Doppler) along the line from you (at the origin) to the target. Radial velocity is the projection of the true velocity onto the line‑of‑sight unit vector ek=pk/∥pk∥e_k = p_k /\|p_k\|ek=pk/∥pk∥[[en.wikipedia](https://en.wikipedia.org/wiki/Radial_velocity)]​.

So for each measurement

vr,k=ek⊤v=cos⁡(αk) ∥v∥v\_{r,k} = e_k^\top v = \cos(\alpha_k)\,\|v\|vr,k=ek⊤v=cos(αk)∥v∥

where αk\alpha_kαk is the angle between the true velocity vector and the line of sight at time tkt_ktk.wikipedia+1

In 2‑D, each measurement gives you a linear equation

ek,xvx+ek,yvy=vr,k.e*{k,x} v_x + e*{k,y} v*y = v*{r,k}.ek,xvx+ek,yvy=vr,k.

If you have at least **two measurements with line‑of‑sight vectors that are not collinear**, you get two independent equations in the two unknowns vx,vyv_x, v_yvx,vy. Solving them yields the full velocity vector (direction and magnitude).

Example with two measurements:

[e1,xe1,ye2,xe2,y][vxvy]=[vr,1vr,2].\begin{bmatrix} e*{1,x} & e*{1,y}\\ e*{2,x} & e*{2,y} \end{bmatrix} \begin{bmatrix} v*x\\ v_y \end{bmatrix} = \begin{bmatrix} v*{r,1}\\ v\_{r,2} \end{bmatrix}.[e1,xe2,xe1,ye2,y][vxvy]=[vr,1vr,2].

If the 2×22\times22×2 matrix of eke_kek is invertible (the two sight lines differ), then

v=[vxvy]=[e1,xe1,ye2,xe2,y]−1[vr,1vr,2].v = \begin{bmatrix} v*x\\ v_y \end{bmatrix} = \begin{bmatrix} e*{1,x} & e*{1,y}\\ e*{2,x} & e*{2,y} \end{bmatrix}^{-1} \begin{bmatrix} v*{r,1}\\ v\_{r,2} \end{bmatrix}.v=[vxvy]=[e1,xe2,xe1,ye2,y]−1[vr,1vr,2].

With more than two measurements (over‑determined system), you can compute a least‑squares estimate of vvv.

---

## When it does not work

- If all measurements are taken while the target is nearly along the **same bearing**, then all eke_kek are parallel and the system collapses to essentially one equation; only the radial component is observable, and the tangential component remains ambiguous.hess.copernicus+1
- A single monostatic radar at one instant can never see the sideways component; you need either motion of the target that rotates the line of sight over time (your case) or multiple spatially separated radars/baselines.vut+1

So: with a noise‑free stationary radar, known positions from each ping, and at least two significantly different lines of sight to a constant‑velocity target, you can reconstruct the true 2‑D velocity vector uniquely by solving the linear system above.
