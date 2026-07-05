# Parametric Curve Parameter Estimation

Finding $\theta$, $M$, and $X$ from 1500 unordered points on a rotated parametric curve (`xy_data.csv`).

---

## Answer

| Parameter | Value |
|-----------|-------|
| $\theta$ | **0.5236 rad (30°)** |
| $X$ | **55.0** |
| $M$ | **0.030** |

### Desmos equation

Check the graph at [https://www.desmos.com/calculator/d48qtlslnr](https://www.desmos.com/calculator/d48qtlslnr):

```
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,\ 42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

![Desmos verification](desmos_graph.png)

---

## Problem

Each point in the CSV satisfies:

$$x = t\cos\theta - e^{M|t|}\sin(0.3t)\sin\theta + X$$

$$y = 42 + t\sin\theta + e^{M|t|}\sin(0.3t)\cos\theta$$

for some unknown $t_i \in (6, 60)$. The rows are **not ordered** by $t$.

### Initial notes

![Handwritten problem statement](HandwrittenPS.jpeg)

*Rewriting the equation in matrix form to understand the rotation structure.*

---

## First attempt — brute force

My first thought was: if there are only three unknowns, just grid-search them.

I set up a 3D grid over $\theta \in [20°, 40°]$, $M \in [-0.05, 0.05]$, $X \in [45, 65]$, sampled the curve densely for each combination, and scored each triple by how close the data points were to the curve (mean minimum L1). This is `brute_force_search.py`.

It actually worked — pointed straight to $\theta \approx 30°$, $X \approx 55$, $M \approx 0.03$. But the grid was coarse (steps of 2° in $\theta$, 5 in $X$) and it had no way to enforce the $t \in [6, 60]$ constraint or assign which $t_i$ belongs to which point. Finer grids would be slow, and there was no real precision guarantee.

So I used brute force to confirm the neighbourhood, then built a proper pipeline to nail down the exact values.

---

## Things I tried before getting there

**Line-fitting for θ and X**

My first clean idea was: if you ignore the sinusoidal oscillation, the curve's "spine" is just a straight line rotated by $\theta$ and passing through $(X, 42)$. So I fit a line $y = mx + c$ to the 1500 points using OLS, read off $\theta = \arctan(m)$, and solved for $X$ from the intercept.

The slope part was fine — gave $\theta \approx 27.8°$, close enough. But extracting $X$ from the intercept was the problem. The intercept tells you where the line hits $x = 0$, which corresponds to $t = 0$ in the original curve. Our data only covers $t \in (6, 60)$, so $t = 0$ is completely outside the observed range. Extrapolating that far out amplified any small angle error into a large $X$ error — I got $X \approx 131$, well outside the valid range of $(0, 100)$.

The fix was obvious in hindsight: instead of solving at $t = 0$, use a point inside the actual data range (like the centroid at $t \approx 33$). But by the time I worked this out I had already switched to the unrotation approach, which doesn't have this extrapolation problem at all.

**Zero-crossing check (as a visual validation)**

As a sanity check on the final answer, I also looked at where $\sin(0.3t) = 0$ — i.e. $t = n\pi/0.3$. At those points $v = 0$ regardless of $M$, so the curve must sit exactly on its own centerline with no sinusoidal contribution. Overlaying these crossing points on the scatter confirmed they line up perfectly on the rotated axis at $\theta = 30°$, $X = 55$ — a clean independent check that needed no fitting at all.

---



## What I did (refined approach)

The main challenge is that each point has its own hidden $t_i$ — you can't just run a standard regression.

**Step 1 — PCA for a starting guess**
PCA on the point cloud gives the spine direction → $\theta \approx 28.5°$.

**Step 2 — Grid search over $(\theta, X)$**
For each candidate pair, I unrotate the points and check what fraction of $u_i$ fall in $[6, 60]$. The best result was $\theta = 29.5°$, $X = 55$ with 99.9% of points in range.

**Step 3 — Fit $M$**
After unrotating, the $v$ values should follow $e^{M|t|}\sin(0.3t)$. A simple 1D bounded minimisation gives $M \approx 0.0294$.

**Step 4 — EM refinement**
Alternate between assigning the nearest $t_i$ to each point and re-fitting $(\theta, X, M)$ with least squares. After 5 rounds RMSE dropped from ~0.18 to ~0.016, converging to the final values above.

**Step 5 — L1 check**
Sampled 500 points uniformly over $t \in [6, 60]$, found the nearest data point to each, and computed Manhattan distance. Mean L1 = **0.0263**.

Also did a brute-force 3D grid first as a sanity check — it landed on the same $(30°, 55, 0.03)$.

---

## Figures

![Data scatter](plots/data_scatter.png)

*Raw point cloud — clearly a rotated, oscillating curve.*

![Canonical frame](plots/canonical_uv.png)

*After unrotating by $\theta$: the sine-wave envelope with growing amplitude is clearly visible.*

![M fit](plots/fit_M_vt.png)

*Fitted $v$ vs $t$ with $M = 0.0294$.*

---

## How to run

```bash
pip install numpy pandas matplotlib scipy
python plot_data.py
python pca_theta.py
python grid_search_theta_X.py
python fit_M.py
python refine_parameters.py
python l1_metric.py
```

Brute-force baseline:
```bash
python brute_force_search.py
```

---

## Files

| File | Purpose |
|------|---------|
| `xy_data.csv` | Input data |
| `curve_model.py` | Curve math, L1 metric |
| `brute_force_search.py` | 3D grid baseline |
| `pca_theta.py` | Initial $\theta$ estimate |
| `grid_search_theta_X.py` | Constraint-based grid search |
| `fit_M.py` | Fit $M$ from unrotated data |
| `refine_parameters.py` | EM refinement loop |
| `l1_metric.py` | Final L1 evaluation |
