import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize_scalar
from curve_model import unrotate_deg, v_model

THETA_DEG = 29.5
X = 55.0

Path("plots").mkdir(exist_ok=True)
df = pd.read_csv("xy_data.csv")
u, v = unrotate_deg(df.x.values, df.y.values, THETA_DEG, X)
t = np.clip(u, 6, 60)

res = minimize_scalar(
    lambda M: np.mean((v - v_model(t, M)) ** 2),
    bounds=(-0.05, 0.05),
    method="bounded",
)
M_hat = res.x

print(f"M = {M_hat:.6f}, MSE(v) = {res.fun:.6f}")

tt = np.linspace(6, 60, 500)
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(t, v, s=5, alpha=0.4, label="unrotated data")
ax.plot(tt, v_model(tt, M_hat), "r-", lw=2, label=f"model, M={M_hat:.5f}")
ax.set_xlabel("t")
ax.set_ylabel("v")
ax.legend()
fig.tight_layout()
fig.savefig("plots/fit_M_vt.png", dpi=120)
print("Saved plots/fit_M_vt.png")
plt.show()
