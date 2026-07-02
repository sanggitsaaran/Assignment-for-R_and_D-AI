import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from curve_model import Y_OFFSET, curve_xy, v_model

df = pd.read_csv("xy_data.csv")
x, y = df.x.values, df.y.values

theta0, X0, M0 = np.radians(29.5), 55.0, 0.0


def assign_t(theta, X, M):
    t_grid = np.linspace(6, 60, 800)
    xg, yg = curve_xy(t_grid, theta, X, M)
    out = np.empty(len(x))
    for i in range(len(x)):
        d2 = (xg - x[i]) ** 2 + (yg - y[i]) ** 2
        out[i] = t_grid[np.argmin(d2)]
    return out


def residuals(p, t_assign):
    theta, X, M = p
    c, s = np.cos(theta), np.sin(theta)
    u, v = t_assign, v_model(t_assign, M)
    x_pred = c * u - s * v + X
    y_pred = s * u + c * v + Y_OFFSET
    return np.concatenate([x - x_pred, y - y_pred])


params = np.array([theta0, X0, M0])
for it in range(5):
    t_assign = assign_t(*params)
    res = least_squares(
        residuals,
        params,
        args=(t_assign,),
        bounds=([0, 0, -0.05], [np.radians(50), 100, 0.05]),
    )
    params = res.x
    rmse = np.sqrt(np.mean(res.fun ** 2))
    print(
        f"iter {it + 1}: theta={np.degrees(params[0]):.4f} deg, "
        f"X={params[1]:.4f}, M={params[2]:.6f}, RMSE={rmse:.5f}"
    )

theta, X, M = params
print(f"\ntheta = {theta:.6f} rad ({np.degrees(theta):.4f} deg)")
print(f"X     = {X:.6f}")
print(f"M     = {M:.6f}")
