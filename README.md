# Parametric Curve Parameter Estimation

Estimate rotation \(\theta\), shape parameter \(M\), and translation \(X\) from 1500 unordered points on a rotated parametric curve (`xy_data.csv`).

---

## Results

| Parameter | Value | Range |
|-----------|-------|-------|
| \(\theta\) | **0.5236 rad** (**30.00°**) | \((0°, 50°)\) |
| \(X\) | **55.0000** | \((0, 100)\) |
| \(M\) | **0.0300** | \((-0.05, 0.05)\) |

**Desmos equation** ([calculator link](https://www.desmos.com/calculator/rfj91yrxob)):

```latex
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,\ 42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

![Desmos overlay](desmos_graph.png)

The fitted curve matches the data: diagonal spine, growing oscillation amplitude, and endpoints near \((60, 46)\) and \((110, 70)\).

---

## Problem

Each point satisfies:

\[
\begin{pmatrix} x \\ y \end{pmatrix}
=
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} t \\ e^{M|t|}\sin(0.3t) \end{pmatrix}
+
\begin{pmatrix} X \\ 42 \end{pmatrix}
\]

Each point has a latent \(t_i \in (6, 60)\). CSV row order does not correspond to \(t\) (Spearman correlation between row index and estimated \(t\) is \(-0.05\); 759 of 1499 rank pairs are inverted).

---

## Why this is hard — and why naive approaches fail

There are only three named unknowns, but each of 1500 points carries its own latent \(t_i\), and the CSV gives no correspondence between row index and \(t\). **Standard regression fails** because \(x\) and \(y\) share a rotation matrix — they cannot be fitted independently. **Fitting against row order fails** because rows are not sorted by \(t\). **A single `curve_fit` on three globals** still needs to know which \(t_i\) belongs to which point; without that, the optimizer has no consistent target.

Our first attempt was a **coarse 3D brute-force grid** over \((\theta, M, X)\), scoring each triple by distance to the data. That correctly locates the parameter basin but searches blindly: it does not enforce \(t \in [6,60]\), does not assign latent \(t_i\) explicitly, and scales poorly as grid resolution increases. We moved to a **geometry-first pipeline** that exploits the rotation structure — unrotate the cloud, read off \(t\) from the canonical coordinate, fit \(M\), then refine by alternating \(t_i\) assignment with global parameter updates.

---

## Approach

### Initial idea: brute-force search

`brute_force_search.py` grids \(\theta \in [20°, 40°]\), \(M \in [-0.05, 0.05]\), \(X \in [45, 65]\) and picks the triple minimizing mean minimum L1 from each **data point** to a densely sampled curve.

### Key structural insight

Define \(u = t\) and \(v = e^{M|t|}\sin(0.3t)\). Then:

\[
\begin{pmatrix} x - X \\ y - 42 \end{pmatrix} = R(\theta) \begin{pmatrix} u \\ v \end{pmatrix}
\]

The observed cloud is a **rigid transform** of the canonical curve \((t,\; e^{M|t|}\sin(0.3t))\). Inverse rotation exposes \(u_i \approx t_i\) without using row order.

### Refined pipeline

```mermaid
flowchart LR
    A[plot_data] --> B[pca_theta]
    B --> C[grid_search_theta_X]
    C --> D[unrotate_plot]
    D --> E[fit_M]
    E --> F[refine_parameters]
    F --> G[l1_metric]
```

| Step | Script | Role |
|------|--------|------|
| 1 | `plot_data.py` | Confirm elongated rotated curve |
| 2 | `pca_theta.py` | PCA spine direction → \(\theta \approx 28.5°\) |
| 3 | `grid_search_theta_X.py` | Search \((\theta, X)\); maximize fraction of \(u_i \in [6,60]\) |
| 4 | `unrotate_plot.py` | Validate canonical \((u,v)\) frame |
| 5 | `fit_M.py` | 1D bounded fit of \(v\) vs \(t\) |
| 6 | `refine_parameters.py` | Alternate \(t_i\) assignment and \((\theta, X, M)\) least squares |
| 7 | `l1_metric.py` | L1 self-check on uniform \(t\) samples |

Shared curve math lives in `curve_model.py`.

### Why not brute force alone?

Brute force finds the right neighborhood but treats the problem as opaque 3D search. The refined pipeline adds: (1) a **constraint-driven** \((\theta, X)\) search via the known \(t\)-range, (2) explicit **latent \(t_i\) assignment** in the EM loop, and (3) **sub-degree parameter precision** via iterative least squares. On the assignment's uniform-\(t\) L1 metric (see below), both methods land at essentially the same error — the refined pipeline earns its keep through principled decomposition and precise convergence, not a dramatically lower L1 at this grid resolution.

---

## Method detail

### Grid-search score (`grid_search_theta_X.py`)

For each \((\theta, X)\), unrotate all points and compute:

\[
\text{score} = \frac{1}{N}\sum_i \mathbf{1}[u_i \in [6,60]] - 0.02\,|\mathrm{range}(u) - 54| - 0.01\,\mathrm{std}(v)
\]

| Rank | \(\theta\) (°) | \(X\) | Score |
|------|----------------|-------|-------|
| 1 | 29.5 | 55.0 | 0.9752 |
| 2 | 30.0 | 55.0 | 0.9748 |
| 3 | 29.0 | 55.0 | 0.9741 |

At the optimum, 99.93% of points have \(u \in [6, 60]\) and \(\mathrm{range}(u) = 53.99 \approx 54\).

### EM refinement (`refine_parameters.py`)

- **E-step:** assign \(t_i = \arg\min_{t \in [6,60]} \|(x_i,y_i) - \gamma(t)\|_2^2\)
- **M-step:** minimize \(\sum_i \|(x_i,y_i) - \gamma(t_i)\|_2^2\) over \((\theta, M, X)\)

Five iterations reduce RMSE from 0.179 to **0.0165**.

### L1 grading metric (`l1_metric.py`)

Both `brute_force_search.py` and `l1_metric.py` call the same function `uniform_t_l1` in `curve_model.py`: sample \(t\) uniformly on \([6, 60]\) (500 points), evaluate \(\gamma(t)\), find the nearest data point, and sum Manhattan distance:

\[
d_j = |x_j - x_{i(j)}| + |y_j - y_{i(j)}|
\]

| Method | Best \((\theta, X, M)\) | Uniform-\(t\) mean L1 |
|--------|-------------------------|------------------------|
| Coarse brute-force grid | 30°, 55, 0.030 | 0.0262 |
| Refined pipeline | 30.00°, 55.000, 0.030 | **0.0263** |

The two methods agree on parameters and L1 to three decimal places. Brute force's internal search objective (data → curve, 0.0512) uses a different sampling direction and is not comparable to the uniform-\(t\) metric above.

| Statistic (refined) | Value |
|---------------------|-------|
| Mean L1 | **0.0263** |
| Median L1 | 0.0182 |
| Max L1 | 0.3488 (at \(t \approx 55.6\)) |

The max L1 is \(\approx 13\times\) the mean. It occurs in the curve interior near \(t \approx 55.6\), not at the endpoints — the region where the scatter plot shows a visible gap in point density (around \(x \approx 105\)). Nearest-neighbor matching degrades where data thins along the curve; this is a coverage artifact, not evidence of a wrong \((\theta, M, X)\).

---

## Figures

**Input data** (`plot_data.py`):

![Data scatter](plots/data_scatter.png)

**Canonical frame** after inverse rotation (`unrotate_plot.py`):

![Canonical u,v](plots/canonical_uv.png)

**Shape fit** \(v\) vs \(t\) (`fit_M.py`):

![v vs t](plots/fit_M_vt.png)

**Desmos verification** — parametric curve with estimated parameters:

![Desmos](desmos_graph.png)

---

## Validation

| Check | Evidence |
|-------|----------|
| \(\theta\) convergence | PCA 28.5° → grid 29.5° → refine 30.00° |
| \(t\)-range constraint | 99.93% of \(u_i \in [6,60]\) after unrotation |
| \(M > 0\) | Envelope ratio \(e^{0.03 \cdot 54} \approx 5\times\) matches growing oscillations in data |
| Brute force vs refined | Same basin on uniform-\(t\) L1; refined gives precise params via EM |
| Desmos overlay | Curve passes through observed point cloud (see above) |

---

## Reproduce

**Requirements:** Python 3, `numpy`, `pandas`, `matplotlib`, `scipy`

```bash
python plot_data.py
python pca_theta.py
python grid_search_theta_X.py
python unrotate_plot.py
python fit_M.py
python refine_parameters.py
python l1_metric.py
```

Initial brute-force baseline (reports both search objective and uniform-\(t\) L1):

```bash
python brute_force_search.py
```

---

## File layout

```
├── xy_data.csv
├── curve_model.py
├── brute_force_search.py
├── plot_data.py
├── pca_theta.py
├── grid_search_theta_X.py
├── unrotate_plot.py
├── fit_M.py
├── refine_parameters.py
├── l1_metric.py
├── desmos_graph.png
├── plots/
│   ├── data_scatter.png
│   ├── canonical_uv.png
│   └── fit_M_vt.png
└── README.md
```
