import numpy as np

Y_OFFSET = 42.0

THETA = 0.523588
X = 54.999639
M = 0.029999


def curve_xy(t, theta=THETA, X=X, M=M):
    t = np.asarray(t)
    v = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    c, s = np.cos(theta), np.sin(theta)
    return c * t - s * v + X, s * t + c * v + Y_OFFSET


def unrotate(x, y, theta_rad, X, y_off=Y_OFFSET):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    dx, dy = x - X, y - y_off
    return c * dx + s * dy, -s * dx + c * dy


def unrotate_deg(x, y, theta_deg, X, y_off=Y_OFFSET):
    return unrotate(x, y, np.radians(theta_deg), X, y_off)


def v_model(t, M):
    return np.exp(M * np.abs(t)) * np.sin(0.3 * t)


def uniform_t_l1(theta, X, M, x_data, y_data, n_samples=500):
    from scipy.spatial import cKDTree

    t = np.linspace(6, 60, n_samples)
    x_pred, y_pred = curve_xy(t, theta, X, M)
    tree = cKDTree(np.column_stack([x_data, y_data]))
    _, idx = tree.query(np.column_stack([x_pred, y_pred]), k=1)
    l1 = np.abs(x_pred - x_data[idx]) + np.abs(y_pred - y_data[idx])
    return t, l1
