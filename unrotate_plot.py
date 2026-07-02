import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from curve_model import unrotate_deg

THETA_DEG = 29.5
X = 55.0

Path("plots").mkdir(exist_ok=True)
df = pd.read_csv("xy_data.csv")
u, v = unrotate_deg(df.x.values, df.y.values, THETA_DEG, X)

print(f"theta={THETA_DEG} deg, X={X}")
print(f"u: [{u.min():.3f}, {u.max():.3f}], frac in [6,60]={np.mean((u >= 6) & (u <= 60)):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(df.x, df.y, s=5, alpha=0.5)
axes[0].set_title("(x, y)")
axes[0].axis("equal")

order = np.argsort(u)
axes[1].scatter(u, v, s=5, alpha=0.5)
axes[1].plot(u[order], v[order], "r-", lw=0.5, alpha=0.5)
axes[1].axvline(6, ls="--", c="gray", lw=0.8)
axes[1].axvline(60, ls="--", c="gray", lw=0.8)
axes[1].set_title("(u, v) canonical frame")
axes[1].set_xlabel("u = t")
axes[1].set_ylabel("v")

fig.tight_layout()
fig.savefig("plots/canonical_uv.png", dpi=120)
print("Saved plots/canonical_uv.png")
plt.show()
