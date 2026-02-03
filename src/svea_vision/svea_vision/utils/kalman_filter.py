# Not a ROS-node, only Kalman Filter
from filterpy.kalman import KalmanFilter
import numpy as np
from math import sin, cos

# KalmanFilters performance depends on the correct setting of parameters
# like the process noise, measurement noise, and the initial state estimate.


# KalmanFilter module implements the linear Kalman filter in both
# an object oriented and procedural form.
class KF(KalmanFilter):
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
        self.dt = 1 / frequency_of_measurements

        process_variance = 0.03
        covariance = 2
        measurement_variance = 1  # Actual camera measurement noise (TBD)
        v_measurement_var = 0.003  # Assumed small
        phi_measurement_var = 0.003  # Assumed small

        # Specify/initialize the Kalman parameters
        self.x = np.array([*init_pos, init_v, init_phi])

        # Control model (based on previous locations)
        # State Transition matrix F to predict the state in the next time period (epoch)
        self.F = np.array(
            [
                [1, 0, self.dt * cos(self.x[3]), 0],
                [0, 1, self.dt * sin(self.x[3]), 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Process uncertainty/noise
        # Covariance matrix Q specifies the process covariance. In Bayesian terms, this
        # prediction is called the *prior*, which you can think as the estimate
        # prior to incorporating the measurement.
        self.Q = process_variance * np.eye(
            len(self.x), len(self.x)
        )  # Process noise covariance matrix

        # Covariance matrix
        self.P = covariance * np.eye(
            len(self.x), len(self.x)
        )  # covariance matrix of state x
        self.R = np.array(
            [
                [measurement_variance, 0, 0, 0],  # Measurement noise covariance matrix
                [0, measurement_variance, 0, 0],
                [0, 0, v_measurement_var, 0],
                [0, 0, 0, phi_measurement_var],
            ]
        )

        # Measurement model
        self.H = np.eye(len(self.x), len(self.x))
    
    def predict(self):
        """Predict the next state."""
        # Update the state transition matrix F based on the current heading (phi)
        self.F[0, 2] = self.dt * cos(self.x[3])
        self.F[1, 2] = self.dt * sin(self.x[3])
        return super().predict()