import numpy as np
import pandas as pd
from curve_model import Y_OFFSET

df = pd.read_csv("xy_data.csv")
x, y = df.x.values, df.y.values


def score(theta_deg, X):
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    dx, dy = x - X, y - Y_OFFSET
    u = c * dx + s * dy
    v = -s * dx + c * dy
    frac = np.mean((u >= 6) & (u <= 60))
    span_pen = abs(u.max() - u.min() - 54.0)
    return frac - 0.02 * span_pen - 0.01 * np.std(v), u


best = (-np.inf, None, None, None)
rows = []
for theta_deg in np.arange(5, 50.5, 0.5):
    for X in np.arange(0, 100.5, 1.0):
        sc, u = score(theta_deg, X)
        rows.append((sc, theta_deg, X))
        if sc > best[0]:
            best = (sc, theta_deg, X, u)

sc, theta_best, X_best, u = best
rows.sort(reverse=True)

print(f"Best: theta={theta_best:.2f} deg, X={X_best:.2f}, score={sc:.4f}")
print(f"u in [6,60]: {np.mean((u >= 6) & (u <= 60)):.4f}, range=[{u.min():.3f}, {u.max():.3f}]")
print("Top 5:")
for sc, th, Xc in rows[:5]:
    print(f"  score={sc:.4f}  theta={th:.1f}  X={Xc:.1f}")
