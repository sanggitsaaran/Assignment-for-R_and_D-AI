import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("plots").mkdir(exist_ok=True)

df = pd.read_csv("xy_data.csv")
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df.x, df.y, s=5, alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Input point cloud")
ax.axis("equal")
fig.tight_layout()
fig.savefig("plots/data_scatter.png", dpi=120)
print("Saved plots/data_scatter.png")
plt.show()
