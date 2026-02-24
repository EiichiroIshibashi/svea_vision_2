# Not a ROS-node, only Kalman Filter
from filterpy.kalman import KalmanFilter
import numpy as np
from math import sin, cos, 

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

        return self.x

    def update(self, z, R=None, H=None):
        z = np.asarray(z, dtype=float).copy()
        z[2] = max(0.0, float(z[2]))

        # Keep heading innovation short to avoid +/-pi discontinuity shocks
        predicted_phi = self._wrap_angle(float(self.x[3]))
        measured_phi = self._wrap_angle(float(z[3]))
        phi_error = self._wrap_angle(measured_phi - predicted_phi)
        z[3] = predicted_phi + phi_error
        
        super().update(z, R=R, H=H)

        # clamp + warp
        self.x[2] = max(0.0, float(self.x[2]))
        self.x[3] = self._wrap_angle(float(self.x[3]))

        return self.x
