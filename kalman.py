"""Generic constant-velocity Kalman filter (n measured quantities, 2n states)."""
import numpy as np


class KalmanCV:
    def __init__(self, z0, q=1.0, r=1.0, vel_var=100.0):
        z0 = np.asarray(z0, dtype=float)
        self.n = len(z0)
        n = self.n
        self.x = np.zeros(2 * n)
        self.x[:n] = z0
        self.P = np.eye(2 * n) * r
        self.P[n:, n:] *= vel_var
        self.q = q
        self.r = r
        self.updates = 0

    def predict(self, dt):
        n = self.n
        F = np.eye(2 * n)
        F[:n, n:] = np.eye(n) * dt
        I = np.eye(n)
        Q = self.q * np.block([[I * dt ** 3 / 3, I * dt ** 2 / 2],
                               [I * dt ** 2 / 2, I * dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, r_diag=None):
        n = self.n
        z = np.asarray(z, dtype=float)
        H = np.zeros((n, 2 * n))
        H[:, :n] = np.eye(n)
        R = np.diag(r_diag) if r_diag is not None else np.eye(n) * self.r
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(2 * n) - K @ H) @ self.P
        self.updates += 1

    @property
    def pos(self):
        return self.x[:self.n]

    @property
    def vel(self):
        return self.x[self.n:]
