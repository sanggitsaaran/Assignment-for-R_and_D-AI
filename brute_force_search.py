import numpy as np
import pandas as pd
from curve_model import curve_xy, uniform_t_l1

df = pd.read_csv("xy_data.csv")
x, y = df.x.values, df.y.values

theta_grid = np.arange(20, 40.5, 2.0)
M_grid = np.linspace(-0.05, 0.05, 6)
X_grid = np.arange(45, 66, 5)
t_dense = np.linspace(6, 60, 400)

best = (np.inf, None)
for theta_deg in theta_grid:
    theta = np.radians(theta_deg)
    for M in M_grid:
        for X in X_grid:
            xg, yg = curve_xy(t_dense, theta, X, M)
            err = np.mean([np.min(np.abs(xg - x[i]) + np.abs(yg - y[i])) for i in range(len(x))])
            if err < best[0]:
                best = (err, theta_deg, X, M)

search_err, theta_deg, X, M = best
theta = np.radians(theta_deg)
_, l1 = uniform_t_l1(theta, X, M, x, y)

print("Coarse 3D brute-force grid:")
print(f"  best (theta, X, M) = ({theta_deg:.1f} deg, {X:.1f}, {M:.4f})")
print(f"  search objective (data->curve, mean min L1): {search_err:.4f}")
print(f"  uniform-t L1 (same metric as l1_metric.py):  {l1.mean():.6f}")
