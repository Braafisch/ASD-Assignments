"""
This is a class for simulating the motion of a vehicle.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


dT = 0.1  # [s] time difference
L = 2.9  # [m] Wheel base of vehicle
veh_dim_x, veh_dim_y = 4, 1.9  # [m] size of vehicle (length, width)
max_steer = np.radians(30.0)  # [rad] max steering angle
max_ax = 2  # [m/ss] max (positive) acceleration
min_ax = -10  # [m/ss] max decceleration (=min negative acceleration)

# sluggish vehicle (only needed for optional excercise):
m = 1800  # [kg] mass
J = 3000  # moment of inertia [kg/m2]
lv = 1.3  # distance COG to front axle [m]
lh = 1.6  # distance COG to rear axle [m]
cav = 2*60000  # lateral tire stiffness front [N/rad]
cah = 2*60000  # lateral tire stiffness rear [N/rad]


def normalize_angle(angle):
    """ Normalize an angle to [-pi, pi]. """
    return (angle + np.pi) % (2 * np.pi) - np.pi


class State():
    """
    Class representing the state of a vehicle.

    :var t: (float) current time
    :var x: (float) x-coordinate
    :var y: (float) y-coordinate
    :var yaw: (float) yaw angle
    :var v: (float) speed
    :var model_type: (string) "kinematic" or "dynamic" (bicyle model) 

    For the one track dynamics model, you additionally need two more state components:

    :var beta: (float) slip angle
    :var dyaw_dt: (float) time derivative of slip angle
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, t=0.0, model_type="dynamic"):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.t = t
        self.model_type = model_type
        assert self.model_type == "static" or self.model_type == "dynamic", "Model type should either be 'static' or 'dynamic'"

        self.beta = 0
        self.dyaw_dt = 0

    def kinematic_model(self, state, t, acceleration, delta):
        """Kinematic vehicle model.
        This function is to be used in odeint and has 
        form "dstate_dt = f(state,t)". 
        """
        x, y, yaw, v = state

        dx_dt = v*np.cos(yaw)
        dy_dt = v*np.sin(yaw)
        dyaw_dt = v*np.tan(delta)/L
        dv_dt = acceleration

        dstate_dt = [dx_dt, dy_dt, dyaw_dt, dv_dt]
        return dstate_dt

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)
        acceleration = np.clip(acceleration, min_ax, max_ax)

        state0 = [self.x, self.y, self.yaw, self.v]
        ti = [self.t, self.t+dT]
        sol = odeint(self.kinematic_model, state0,
                     ti, args=(acceleration, delta))

        self.x, self.y, self.yaw, self.v = sol[1]
        self.yaw = normalize_angle(self.yaw)
        self.t = ti[1]

    def dynamic_model(self, state, t, acceleration, delta):
        """Model for the lateral and yaw dynamics of the bicylce model.

        This function is to be used in odeint and has 
        form "dstate_dt = f(state,t)". 
        """
        x, y, yaw, v, beta, dyaw_dt = state

        dbeta_dt = - (cav+cah)/(m*v)*beta - (1+(cav*lv-cah*lh) /
                                             (m*v**2))*dyaw_dt + cav/(m*v)*delta
        ddyaw_dt2 = - (cav*lv-cah*lh)/J*beta - (cav*lv**2 +
                                                cah*lh**2)/(J*v)*dyaw_dt + cav*lv/J*delta

        dx_dt = v*np.cos(yaw+beta)
        dy_dt = v*np.sin(yaw+beta)
        dyaw_dt = dyaw_dt
        dv_dt = acceleration

        dstate_dt = [dx_dt, dy_dt, dyaw_dt, dv_dt, dbeta_dt, ddyaw_dt2]
        return dstate_dt

    def update_dynamic_model(self, acceleration, delta):
        """
        Update the state of the vehicle (only needed for optional excercise).

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)
        acceleration = np.clip(acceleration, min_ax, max_ax)

        state0 = [self.x, self.y, self.yaw, self.v, self.beta, self.dyaw_dt]
        ti = [self.t, self.t+dT]
        sol = odeint(self.dynamic_model, state0,
                     ti, args=(acceleration, delta))

        self.x, self.y, self.yaw, self.v, self.beta, self.dyaw_dt = sol[1]
        self.yaw = normalize_angle(self.yaw)
        self.beta = normalize_angle(self.beta)
        self.t = ti[1]
