import numpy as np
from scipy.stats import norm

from utils import gaussian_bbox

class ConstantVelocityModel:
    """Constant velocity motion model."""

    def __init__(self, q):
        """Init."""
        self.q = q

    def __call__(self, xprev, Pprev, dT):
        """Step model."""
        x = xprev
        F = np.array([[1, 0, dT, 0],
                       [0, 1, 0, dT],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        Q = np.array([[dT ** 3 / 3, 0,           dT ** 2 / 2, 0],
                      [0,           dT ** 3 / 3, 0,           dT ** 2 / 2],
                      [dT ** 2 / 2, 0,           dT,          0],
                      [0,           dT ** 2 / 2, 0,           dT]]) * self.q
        x = np.matmul(F, xprev)
        P = np.matmul(np.matmul(F, xprev), np.transpose(F)) + Q   # F @ Pprev @ F.T + Q

        return (x, P)

def position_measurement(x):
    """Velocity measurement model."""
    H = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]])
    return (np.matmul(H, x), H)


def velocity_measurement(x):
    """Velocity measurement model."""
    H = np.array([[0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return (np.matmul(H, x), H)

class KalmanFilter():
    """Kalman-filter Class
    """

    def __init__(self, model, x0, P0, H, R):
        """Init."""
        self.model = model
        self.x = x0
        self.P = P0
        self.H = H
        self.R = R
        self.trace = [(x0, P0)]
        self._calc_bbox()

    def __repr__(self):
        """Return string representation of measurement."""
        return "T({}, P)".format(self.x)

    def predict(self, dT):
        """Perform motion prediction."""
        new_x, new_P = self.model(self.x, self.P, dT)
        self.trace.append((new_x, new_P))
        self.x, self.P = new_x, new_P

        self._calc_bbox()

    def correct(self, measurements):
        """Perform correction (measurement) update.
        inputs:
            measurements: 2*1 matrix
        """
        S = np.matmul(np.matmul(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.matmul(np.matmul(self.P, np.transpose(self.H)), np.linalg.pinv(S))
        y = measurements - np.matmul(self.H, self.x)
        self.x += np.matmul(K, y)
        self.P -= np.matmul(np.matmul(K, self.H), self.P)

        # score = dz.T @ SI @ dz / 2.0 + ln(2 * pi * sqrt(det(S)))

        self._calc_bbox()

        # return float(score)

    def _calc_bbox(self, nstd=2):
        """Calculate minimal bounding box approximation."""
        self._bbox = gaussian_bbox(self.x[0:2], self.P[0:2, 0:2])

    def bbox(self):
        """Get minimal bounding box approximation."""
        return self._bbox
