import numpy as np
import pandas as pd
from curve_model import THETA, X, M, uniform_t_l1

N_SAMPLES = 500

df = pd.read_csv("xy_data.csv")
t, l1 = uniform_t_l1(THETA, X, M, df.x.values, df.y.values, N_SAMPLES)

print(f"Uniform t samples: {N_SAMPLES}")
print(f"theta={THETA:.6f}, X={X:.6f}, M={M:.6f}")
print(f"Mean Manhattan L1:   {l1.mean():.6f}")
print(f"Median Manhattan L1: {np.median(l1):.6f}")
print(f"Max Manhattan L1:    {l1.max():.6f}  (at t={t[l1.argmax()]:.2f})")
