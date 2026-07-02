import numpy as np
import pandas as pd

df = pd.read_csv("xy_data.csv")
x, y = df.x.values, df.y.values
pts = np.column_stack([x - x.mean(), y - y.mean()])

_, sv, Vt = np.linalg.svd(pts, full_matrices=False)
d = Vt[0]
candidates = [np.arctan2(d[1], d[0]), np.arctan2(d[1], d[0]) + np.pi]

print("Singular values:", sv)
for th in candidates:
    deg = np.degrees(th) % 360
    if 0 < deg < 50:
        print(f"theta guess: {th:.6f} rad ({deg:.4f} deg)")
