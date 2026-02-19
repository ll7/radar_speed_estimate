Yes, absolutely. You can calculate the exact true velocity (both magnitude and orientation) of the object using two or more noise-free measurements, provided the object has moved enough that the angle of your line of sight has changed.

Your hand-drawn diagram perfectly captures the underlying geometry of this problem. The formulas you wrote, $\cos(\alpha_1) = \frac{v_{r1}}{v}$ and $\cos(\alpha_2) = \frac{v_{r2}}{v}$, are exactly the right starting point.

Here is how you can mathematically solve for the true velocity.

### Method 1: Using Radial Velocity (Doppler) Equations

Since you have the $x$ and $y$ positions, you know the exact line of sight (LOS) angle from the sensor to the object for each measurement.

1. **Find the Line of Sight (LOS) angles:**

   For any measurement $i$ at position $(x_i, y_i)$, the angle $\theta_i$ from the sensor to the object is:

   $$\theta_i = \arctan\left(\frac{y_i}{x_i}\right)$$

   _(Note: This creates a unit vector pointing at the object: $\hat{u}_i = (\cos\theta_i, \sin\theta_i)$)_

2. **Set up the projection equations:**

   The radial velocity $v_{ri}$ you measure is simply the dot product (projection) of the true velocity vector $\vec{v} = (v_x, v_y)$ onto the line of sight.

   $$v_{ri} = v_x \cos(\theta_i) + v_y \sin(\theta_i)$$

3. **Solve the system of linear equations:**

   With two measurements, you have two equations and two unknowns ($v_x$ and $v_y$):

   $$v_{r1} = v_x \cos(\theta_1) + v_y \sin(\theta_1)$$

   $$v_{r2} = v_x \cos(\theta_2) + v_y \sin(\theta_2)$$

   You can write this as a matrix:

   $$\begin{pmatrix} v_{r1} \\ v_{r2} \end{pmatrix} = \begin{pmatrix} \cos\theta_1 & \sin\theta_1 \\ \cos\theta_2 & \sin\theta_2 \end{pmatrix} \begin{pmatrix} v_x \\ v_y \end{pmatrix}$$

   As long as $\theta_1 \neq \theta_2$ (meaning the object hasn't traveled exactly along the exact same straight radial line toward/away from you), this matrix is invertible. You can solve for $v_x$ and $v_y$, which gives you the true velocity vector.

   Once you have $v_x$ and $v_y$:
   - **Magnitude (Speed):** $v = \sqrt{v_x^2 + v_y^2}$
   - **Orientation (Heading):** $\phi = \arctan\left(\frac{v_y}{v_x}\right)$

---

### Method 2: Using Position over Time (Kinematics)

If your radar system also records the **timestamp** ($t_1$ and $t_2$) for each measurement, you actually don't even need the radial velocity to find the true velocity! Because you stated the object has _uniform motion_ (constant velocity), you can calculate it purely from the change in position:

$$v_x = \frac{x_2 - x_1}{t_2 - t_1}$$

$$v_y = \frac{y_2 - y_1}{t_2 - t_1}$$

**Why use Method 1 if Method 2 is so simple?** In real-world (noisy) radar applications, position measurements ($x, y$) often have high noise variance, while Doppler-derived radial velocity ($v_r$) is extremely precise. Therefore, combining multiple high-precision $v_r$ measurements (Method 1) often yields a much more accurate true velocity estimate than just tracking the noisy $(x, y)$ positions over time.

Would you like me to walk through a quick numerical example using dummy values to show how the matrix math works in practice?
