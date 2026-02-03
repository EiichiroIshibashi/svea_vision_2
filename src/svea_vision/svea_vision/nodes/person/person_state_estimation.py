#! /usr/bin/env python3

from collections import deque
import numpy as np
from math import sin, cos, atan2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64
from svea_vision_msgs.msg import StampedObjectPoseArray, PersonState, PersonStateArray
from scipy.optimize import curve_fit
from svea_vision.utils.kalman_filter import KF


# Purpose: To track and predict the state of each object detected by a camera system,
# using a combination of interpolation, Kalman Filter, and curve fitting.


class PersonStatePredictor(Node):
    """Class that estimates the states of each detected object
    (x, y, v, phi) by interpolating the locations up to
    MAX_HISTORY_LEN."""

    THRESHOLD_DIST = 0.5  # TODO: Keep the same person id if the distance is not high between two measurements. Improve threshold
    MAX_HISTORY_LEN = 10  # Used for trajectory prediction and average the frequency of people localization message
    MAX_FRAMES_ID_MISSING = 4  # drop id after certain frames

    person_tracker_dict = dict()
    person_states = dict()
    kf_dict = dict()
    kf_state_tracker = dict()
    time_deque = deque([0],MAX_HISTORY_LEN)

    def __init__(self):
        super().__init__("person_state_estimation")
        self.last_time = None
        self.frequency = 0
        self.counter = 0
        self.total_time = 0
        # Initialize the publisher for Twist messages
        self.pub_kf = self.create_publisher(PersonStateArray, "~/person_states_kf", 10)
        self.start()

    def __listener(self):
        """Subscribes to the topic containing only detected
        persons and applies the function __callback."""
        self.create_subscription(
            StampedObjectPoseArray,
            "/detection_splitter/persons",
            self.__callback,
            10
        )

    def __callback(self, msg):
        """This method is a callback function that is triggered when a message is received.
        It interpolates person locations, stores the states of persons, and publishes
        the estimated person states to a 'person_state_estimation/person_states' topic.
        This implementation keeps sending the states of persons who have
        dropped out of frame, because the person might have dropped randomly.
        
        :param msg: message containing the detected persons
        :return: None"""

        # Increment the message counter for ROS 2 compatibility
        self.counter += 1

        self.__update_frequency(msg)
        personStateArray_msg = PersonStateArray()
        personStateArray_msg.header = msg.header

        for person in msg.objects:
            # Get the person's ID and current location
            person_id = person.object.id
            person_loc = (
                person.pose.pose.position.x,
                person.pose.pose.position.y,
                person.pose.pose.position.z,
            )

            # Check if the new measurement is close to previous stored measurements and
            # set person_id if criteria is met.
            person_id = self.recover_id(person_id, person_loc)

            # Append the current person's location in the dict
            if person_id in self.person_tracker_dict:
                self.person_tracker_dict[person_id].append(person_loc)
            else:
                # Initiate a deque that only can contain the wanted length of historical locations
                self.person_tracker_dict[person_id] = deque(
                    [person_loc], maxlen=self.MAX_HISTORY_LEN
                )

            # Estimate the state when having enough historical locations
            if len(self.person_tracker_dict[person_id]) == self.MAX_HISTORY_LEN:
                # Get the velocity and heading for kalman filter estimation
                v, phi = self.fit(self.person_tracker_dict[person_id])

                # Run the Kalman filter
                if not self.kf_dict.get(person_id):
                    # initial position, velocity and heading
                    self.kf_dict[person_id] = KF(
                        person_id,
                        [person_loc[0], person_loc[1]],
                        v,
                        phi,
                        self.frequency,
                    )
                else:
                    self.kf_dict[person_id].predict()
                    z = [
                        person_loc[0],
                        person_loc[1],
                        v,
                        phi,
                    ]  # measurement - actual observaation
                    self.kf_dict[person_id].update(z)

                kf_state = self.kf_dict[person_id].x  # Kalman filter state estimate

                # Check if the person_id is in the kf_state_tracker, append its (x,y) pos
                # or create new deque list.
                if person_id in self.kf_state_tracker:
                    self.kf_state_tracker[person_id].append((kf_state[0], kf_state[1]))
                else:
                    self.kf_state_tracker[person_id] = deque(
                        [(kf_state[0], kf_state[1])], maxlen=self.MAX_HISTORY_LEN
                    )

                # Use the KF estimate for calculating the velocity and heading
                if len(self.kf_state_tracker[person_id]) == self.MAX_HISTORY_LEN:
                    # If the kf state tracker is large enough, we can fit the new heading and
                    # velocity using the predicted pedestrian locations from the Kalman filter.
                    v, phi = self.fit(self.kf_state_tracker[person_id])
                else:
                    # Otherwise, we just use the original estimate
                    v, phi = kf_state[2], kf_state[3]

                state = PersonState()
                pose = Pose()

                # estimate oreintation using the heading
                euler_angle = np.array(
                    [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]
                )  # rotation matrix in three-dimensional space
                quaternion = rotationMatrixToQuaternion1(
                    euler_angle
                )  #  quaternion matrix of the rotation

                pose.position.x, pose.position.y, pose.position.z = (
                    kf_state[0],
                    kf_state[1],
                    person_loc[2],
                )

                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ) = quaternion

                state.id = person_id
                state.pose = pose  # position and orientation
                state.vx = v * cos(phi)
                state.vy = v * sin(phi)
                state.ax = 0.0
                state.ay = 0.0
                state.counter = self.counter

                # Update the dictionary with {ID: PersonState}
                self.person_states[person_id] = state

        # Cleanup the dictionary of person_states
        self.__clean_up_dict(self.counter)

        # Put the list of personstate in the message and publish it
        personStateArray_msg.personstate = list(self.person_states.values())
        self.pub_kf.publish(personStateArray_msg)

    def __calculate_velocity_heading(self, c0, c1):
        """Coordinate interpolation

        :param c0:   tuple(x0,y0) second last position
        :param c1:   tuple(x1,y1) last position
        :return:   velocity and heading"""

        x1, y1 = np.array(c1)
        x0, y0 = np.array(c0)

        dt = float(self.time_deque[-1] - self.time_deque[-2])

        # Prevent division by zero or invalid dt
        if not np.isfinite(dt) or dt <= 1e-6:
            v = 0.0
            phi = atan2(y1 - y0, x1 - x0) 
            return v, float(phi)

        dp = np.array([x1 - x0, y1 - y0], dtype=float)
        dist = float(np.linalg.norm(dp))

        if not np.isfinite(dist):
            return 0.0, float(atan2(y1 - y0, x1 - x0))

        v = dist / dt
        phi = atan2(y1 - y0, x1 - x0)
        return float(v), float(phi)


    def fit(self, trajectory: deque):
        """Fit the trajectory of the previous positions in order to get a
        better estimate of current velocity and heading. First estimate trajectory values
        through curve fitting, and then calculate velocity and heading.

        :param trajectory:   queue of locations
        :return:   A better estimation of the current velocity and heading"""

        x, y = np.zeros(len(trajectory)), np.zeros(len(trajectory))

        for i, path in enumerate(trajectory):
            x[i], y[i] = path[0], path[1]

        time = np.array(self.time_deque)
        time-=min(time)
        popt_x, *_ = curve_fit(quadratic_func, time, x)
        popt_y, *_ = curve_fit(quadratic_func, time, y)
        ax, bx, cx = popt_x
        ay, by, cy = popt_y

        xs = quadratic_func(time, ax, bx, cx)  # TODO: maybe use next timestep?
        ys = quadratic_func(time, ay, by, cy)
        c0, c1 = (xs[-2], ys[-2]), (xs[-1], ys[-1])
        return self.__calculate_velocity_heading(c0,c1)


    def recover_id(self, person_id, person_loc):
        # TODO: Fix this.
        #   -   Scenario: If 2 people are in the frame and close to each other
        #       and 1 of them is suddenly dropped, then new measurements of the
        #       dropped person would probably be too close to the one that is
        #       still tracked, thus only tracking one person.

        if self.person_tracker_dict:
            return person_id

        min_dist = 1000
        for o_id, o_loc in self.person_tracker_dict.items():
            dist = np.linalg.norm(np.array(person_loc[0:2]) - np.array(o_loc[-1][0:2]))
            if dist < min_dist:
                min_dist = dist
                if min_dist < self.THRESHOLD_DIST:
                    person_id = o_id

        return person_id

    def __clean_up_dict(self, current_count):
        ids_to_drop = []
        for id, state in self.person_states.items():
            c = state.counter
            if current_count - c >= self.MAX_FRAMES_ID_MISSING:
                ids_to_drop.append(id)

        for id in ids_to_drop:
            self.person_states.pop(
                id
            )  # TODO: check why not drop it from person_state_tracker?
            self.kf_dict.pop(id)  # TODO: check why not drop it from kf_state_tracker?

    def __update_frequency(self,msg:StampedObjectPoseArray):
        """
        Function used to update the frequency of the message received on "/detection_splitter/persons".
        The frequency computation is done when MAX_HISTORY_LEN messages are received to filter undesired fluctuations.
        """
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
        if self.last_time is not None:
            time_diff = (current_time - self.last_time)
            self.total_time+=time_diff
            self.time_deque.append(self.time_deque[-1] + time_diff)
            if self.counter == self.MAX_HISTORY_LEN:
                self.frequency = self.MAX_HISTORY_LEN / self.total_time
                self.counter = 0
                self.total_time = 0
        self.last_time = current_time

    def start(self):
        self.__listener()


def quadratic_func(t, a, b, c):
    """This function is used for fitting previous positions.

    :param t: time
    :param a: coefficient
    :param b: coefficient
    :param c: coefficient
    :return: a*t**2 + b*t + c which is how the velocity changes over time.
    """
    return a * t**2 + b * t + c


def rotationMatrixToQuaternion1(m):
    """The `rotationMatrixToQuaternion1` function takes a rotation matrix
    and returns a quaternion representation of the rotation.

    :param m: rotation matrix
    :return: quaternion representation of the rotation"""

    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if t > 0:
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (m[2, 1] - m[1, 2]) * t
        q[1] = (m[0, 2] - m[2, 0]) * t
        q[2] = (m[1, 0] - m[0, 1]) * t

    else:
        i = np.argmax([m[0, 0], m[1, 1], m[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3

        t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k, j] - m[j, k]) * t
        q[j] = (m[j, i] + m[i, j]) * t
        q[k] = (m[k, i] + m[i, k]) * t

    q[1] = -q[1]  # TODO check if this is correct

    return q


def main(args=None):
    rclpy.init(args=args)
    predictions = PersonStatePredictor()
    try:
        rclpy.spin(predictions)
    except KeyboardInterrupt:
        pass
    finally:
        predictions.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
