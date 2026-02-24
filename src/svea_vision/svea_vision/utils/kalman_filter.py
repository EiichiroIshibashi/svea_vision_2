# Not a ROS-node, only Kalman Filter
from filterpy.kalman import KalmanFilter
import numpy as np
from math import sin, cos, pi
from dataclasses import dataclass
from typing import Optional

@dataclass
class KFSnapshot:
    # measurement (z)
    z: Optional[np.ndarray] = None
    # prior (after predict)
    x_pred: Optional[np.ndarray] = None
    p_pred: Optional[np.ndarray] = None
    # posterior (after update)
    x_post: Optional[np.ndarray] = None
    p_post: Optional[np.ndarray] = None
    # innovation + stats (after update)
    innovation: Optional[np.ndarray] = None # y
    S: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    nis: float = float("nan")

# KalmanFilters performance depends on the correct setting of parameters
# like the process noise, measurement noise, and the initial state estimate.


# KalmanFilter module implements the linear Kalman filter in both
# an object oriented and procedural form.
class KF(KalmanFilter):
    MIN_VALID_FREQUENCY_HZ = 1.0
    DEFAULT_FREQUENCY_HZ = 15.0

    def __init__(
        self,
        id,
        init_pos: list,
        init_v: float,
        init_phi: float,
        frequency_of_measurements: float = 14.5,
    ):
        """Kalman Filter implementation that also use the
        id to keep track of separate measurements.
         - The state is: [x,y,v,phi]"""

        super().__init__(
            dim_x=4, dim_z=4
        )  # dimensions of state vector and measurement vector

        # Set class attributes
        self.id = id
        self._set_dt_from_frequency(frequency_of_measurements)

        # Pedestrian-oriented default tuning:
        # - Camera XY is usually less noisy than frame-to-frame velocity/heading estimates
        # - Keep v, phi measurements weakly trusted to prevent violent state flips
        self.sigma_xy_meas = 0.20     # [m]
        self.sigma_v_meas = 15.00      # [m/s]
        self.sigma_phi_meas = 2.0    # [rad]
        self.sigma_v_process = 5.00   # [m/s^2] random walk in speed
        self.sigma_phi_process = 3.0 # [rad/s] random walk in heading

        # Specify/initialize the Kalman parameters
        self.x = np.array([*init_pos, max(0.0, float(init_v)), self._wrap_angle(float(init_phi))], dtype=float)

        # Control model (based on previous locations)
        # State Transition matrix F to predict the state in the next time period (epoch)
        self.F = np.array(
            [
                [1, 0, self.dt * cos(self.x[3]), 0],
                [0, 1, self.dt * sin(self.x[3]), 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        , dtype=float)

        # Covariance matrix
        self.P = np.diag([1.0, 1.0, 4.0, 3.0]).astype(float)
        self.R = np.array(
            [
                [self.sigma_xy_meas**2, 0, 0, 0],  # Measurement noise covariance matrix
                [0, self.sigma_xy_meas**2, 0, 0],
                [0, 0, self.sigma_v_meas**2, 0],
                [0, 0, 0, self.sigma_phi_meas**2],
            ]
        , dtype=float)

        # Measurement model
        self.H = np.eye(len(self.x), len(self.x), dtype=float)
        self._update_process_noise()
        self.debug = KFSnapshot()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + pi) % (2.0 * pi) - pi)

    def _set_dt_from_frequency(self, frequency_of_measurements: float) -> None:
        freq = float(frequency_of_measurements) if np.isfinite(frequency_of_measurements) else 0.0
        if freq < self.MIN_VALID_FREQUENCY_HZ:
            freq = self.DEFAULT_FREQUENCY_HZ
        self.dt = 1.0 / freq

    def _update_process_noise(self) -> None:
        # Position uncertainty grows from uncertainty in (v, phi)
        sigma_x_process = max(0.02, self.dt * self.sigma_v_process)
        sigma_y_process = max(0.02, self.dt * self.sigma_v_process)
        sigma_v_step = max(0.02, self.dt * self.sigma_v_process)
        sigma_phi_step = max(0.03, self.dt * self.sigma_phi_process)
        self.Q = np.diag(
            [
                sigma_x_process**2,
                sigma_y_process**2,
                sigma_v_step**2,
                sigma_phi_step**2,
            ]
        ).astype(float)

    def predict(self, dt: float = None):
        """Predict the next state."""
        if dt is not None and np.isfinite(dt) and dt > 1e-3:
            self.dt = float(dt)

        # Update the state transition matrix F based on the current heading (phi)
        self.F[0, 2] = self.dt * cos(self.x[3])
        self.F[1, 2] = self.dt * sin(self.x[3])
        self._update_process_noise()
        super().predict()
        self.x[2] = max(0.0, float(self.x[2]))
        self.x[3] = self._wrap_angle(float(self.x[3]))

        # Store debug info
        self.debug.x_pred = self.x.copy()
        self.debug.P_pred = self.P.copy()
        return self.x

    def update(self, z, R=None, H=None):
        z = np.asarray(z, dtype=float).copy()
        z[2] = max(0.0, float(z[2]))

        # Keep heading innovation short to avoid +/-pi discontinuity shocks
        predicted_phi = self._wrap_angle(float(self.x[3]))
        measured_phi = self._wrap_angle(float(z[3]))
        phi_error = self._wrap_angle(measured_phi - predicted_phi)
        z[3] = predicted_phi + phi_error

        # Store debug info before update
        self.debug.z = z.copy()
        
        super().update(z, R=R, H=H)

        # clamp + warp
        self.x[2] = max(0.0, float(self.x[2]))
        self.x[3] = self._wrap_angle(float(self.x[3]))

        # -- debug snapshot: posterior
        self.debug.x_post = self.x.copy()
        self.debug.P_post = self.P.copy()

        # innovation / S / K (filterpy sets these on update)
        # y is usually shape (dim_z, 1) or (dim_z,)
        try:
            y = np.asarray(self.y, dtype=float).reshape(-1)
        except Exception:
            y = None

        if y is not None and y.size >= 4:
            # make sure angle innovation is wrapped
            y = y.copy()
            y[3] = self._wrap_angle(float(y[3]))
            self.debug.innovation = y

        try:
            self.debug.S = np.asarray(self.S, dtype=float).copy()
        except Exception:
            self.debug.S = None

        try:
            self.debug.K = np.asarray(self.K, dtype=float).copy()
        except Exception:
            self.debug.K = None

        # NIS = y^T S^-1 y
        try:
            if self.debug.innovation is not None and self.debug.S is not None:
                yv = self.debug.innovation.reshape(-1, 1)
                Sinv = np.linalg.inv(self.debug.S)
                self.debug.nis = float((yv.T @ Sinv @ yv)[0, 0])
            else:
                self.debug.nis = float("nan")
        except Exception:
            self.debug.nis = float("nan")

        return self.x
