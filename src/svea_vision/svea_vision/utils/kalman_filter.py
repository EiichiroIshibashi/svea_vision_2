# Not a ROS-node, only Kalman Filter
import numpy as np
from filterpy.kalman import KalmanFilter
class KF(KalmanFilter):
    MIN_VALID_FREQUENCY_HZ = 1.0
    DEFAULT_FREQUENCY_HZ = 15.0

    def __init__(
        self,
        id,
        init_pos: list,
        init_vx: float = 0.0,
        init_vy: float = 0.0,
        frequency_of_measurements: float = 14.5,
    ):
        """Constant Velocity Kalman filter.
        State: [x, y, vx, vy]
        Measurement: [x, y]
        """
        super().__init__(dim_x=4, dim_z=2)

        self.id = id
        self._set_dt_from_frequency(frequency_of_measurements)

        # Pedestrian-oriented default tuning.
        self.sigma_xy_meas = 0.20  # [m]
        self.sigma_acc_process = 4.00  # [m/s^2]

        self.x = np.array([init_pos[0], init_pos[1], float(init_vx), float(init_vy)], dtype=float)
        self.F = np.eye(4, dtype=float)
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self.P = np.diag([1.0, 1.0, 4.0, 4.0]).astype(float)
        self.R = np.diag([self.sigma_xy_meas**2, self.sigma_xy_meas**2]).astype(float)
        self._update_motion_model()

    def _set_dt_from_frequency(self, frequency_of_measurements: float) -> None:
        freq = float(frequency_of_measurements) if np.isfinite(frequency_of_measurements) else 0.0
        if freq < self.MIN_VALID_FREQUENCY_HZ:
            freq = self.DEFAULT_FREQUENCY_HZ
        self.dt = 1.0 / freq

    def _update_motion_model(self) -> None:
        dt = float(self.dt)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = float(self.sigma_acc_process) ** 2

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        self.Q = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=float,
        )

    def predict(self, dt: float = None):
        if dt is not None and np.isfinite(dt) and dt > 1e-3:
            self.dt = float(dt)
        self._update_motion_model()
        super().predict()
        return self.x

    def update(self, z, R=None, H=None):
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size < 2:
            raise ValueError("Measurement z must contain at least [x, y].")
        z = z[:2].copy()

        super().update(z, R=R, H=H)

        return self.x
