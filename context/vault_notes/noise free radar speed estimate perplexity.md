<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Consider we have a noise free radar sensor and we have for one object with a uniform motion 2 or more sensor measurments. the measurements include the x,y position and the relative velocity towards us, assume we are stationary. can we calculate the true velocity of the measured object with respect to the orientation and magnitude? We only measure the relativ velocity towards us but with more than one measurement can we somehow calculate the true velocity?

Yes, but only if those multiple measurements give you **different viewing angles** to the same constant‑velocity target (i.e., the object has moved so that the line of sight changes). Otherwise the problem is under‑determined.

---

## Geometry and equations

Let the target have constant (unknown) velocity vector $v = (v_x, v_y)$.
At time $t_k$ you know its position $p_k = (x_k, y_k)$ and the measured radial velocity $v_{r,k}$ (Doppler) along the line from you (at the origin) to the target. Radial velocity is the projection of the true velocity onto the line‑of‑sight unit vector $e_k = p_k /\|p_k\|$[^1].

So for each measurement

$$
v_{r,k} = e_k^\top v = \cos(\alpha_k)\,\|v\|
$$

where $\alpha_k$ is the angle between the true velocity vector and the line of sight at time $t_k$.[^2][^1]

In 2‑D, each measurement gives you a linear equation

$$
e_{k,x} v_x + e_{k,y} v_y = v_{r,k}.
$$

If you have at least **two measurements with line‑of‑sight vectors that are not collinear**, you get two independent equations in the two unknowns $v_x, v_y$. Solving them yields the full velocity vector (direction and magnitude).

Example with two measurements:

$$
\begin{bmatrix}
e_{1,x} & e_{1,y}\\
e_{2,x} & e_{2,y}
\end{bmatrix}
\begin{bmatrix}
v_x\\ v_y
\end{bmatrix}
=
\begin{bmatrix}
v_{r,1}\\ v_{r,2}
\end{bmatrix}.
$$

If the $2\times2$ matrix of $e_k$ is invertible (the two sight lines differ), then

$$
v =
\begin{bmatrix}
v_x\\ v_y
\end{bmatrix}
=
\begin{bmatrix}
e_{1,x} & e_{1,y}\\
e_{2,x} & e_{2,y}
\end{bmatrix}^{-1}
\begin{bmatrix}
v_{r,1}\\ v_{r,2}
\end{bmatrix}.
$$

With more than two measurements (over‑determined system), you can compute a least‑squares estimate of $v$.

---

## When it does not work

- If all measurements are taken while the target is nearly along the **same bearing**, then all $e_k$ are parallel and the system collapses to essentially one equation; only the radial component is observable, and the tangential component remains ambiguous.[^1][^2]
- A single monostatic radar at one instant can never see the sideways component; you need either motion of the target that rotates the line of sight over time (your case) or multiple spatially separated radars/baselines.[^3][^4]

So: with a noise‑free stationary radar, known positions from each ping, and at least two significantly different lines of sight to a constant‑velocity target, you can reconstruct the true 2‑D velocity vector uniquely by solving the linear system above.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://en.wikipedia.org/wiki/Radial_velocity

[^2]: https://hess.copernicus.org/articles/13/651/2009/hess-13-651-2009.pdf

[^3]: https://www.vut.cz/www_base/zav_prace_soubor_verejne.php?file_id=64454

[^4]: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2016.0024

[^5]: image.jpg

[^6]: https://wx.erau.edu/faculty/mullerb/Wx365/Doppler_velocity/radial_velocity.html

[^7]: https://www.radartutorial.eu/11.coherent/Radial Speed.en.html

[^8]: https://www.weather.gov/Jetstream/vel

[^9]: http://ww2010.atmos.uiuc.edu/(Gh)/wwhlpr/radial_velocity.rxml

[^10]: https://blog.truegeometry.com/calculators/How_do_you_compute_the_range_and_velocity_of_a_target_using_Pulse_Doppler_processing_calculation.html

[^11]: https://www.mathworks.com/help/radar/ug/generate-clutter-and-target-returns-for-mti-radar.html

[^12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3927511/

[^13]: http://ww2010.atmos.uiuc.edu/(Gh)/guides/rs/rad/basics/dpmod.rxml

[^14]: https://arxiv.org/html/2502.01147v1

[^15]: https://publications.sto.nato.int/publications/STO Educational Notes/RTO-EN-SET-119-2009/EN-SET-119(2009)-03.pdf

[^16]: https://www.mathworks.com/help/radar/ug/doppler-estimation.html
