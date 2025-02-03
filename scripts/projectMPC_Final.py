import os
import sys

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from PIL import Image

# Model params
L1 = 1
L2 = 1
LC1 = L1 / 2
LC2 = L2 / 2
M1 = 1
M2 = 1
G = 9.81
TS = 0.1

## Moment of inertias about respective pivots
I1 = (1 / 3) * M1 * L1**2
I2 = (1 / 3) * M2 * L2**2
X0 = np.array([0, 0, 0, 0])
XN = np.array([np.pi, 0, 0, 0])
U_MIN = np.array([-10, -10])
U_MAX = np.array([10, 10])
OMEGA_MAX = np.array([2.5, 2.5])
OMEGA_MIN = np.array([-2.5, -2.5])


"""
# Objecta Near
center = (1.75, 1.75)
radius = 0.5

center2 = (1.75, -1.75)
radius2 = 0.5
NUM_OF_POINTS_ALONG_ARM = 5

"""
# Objecta Near
center = (1.45, 1.45)
radius = 0.5

center2 = (1.45, -1.45)
radius2 = 0.5
NUM_OF_POINTS_ALONG_ARM = 5


# Dynamics
def dynamics(x, u):
    theta = x[0:2]
    omega = x[2:4]
    tao = u

    A = (
        tao[0]
        + 2 * M2 * L1 * LC2 * ca.sin(theta[1]) * omega[0] * omega[1]
        + M2 * L1 * LC2 * ca.sin(theta[1]) * omega[1] ** 2
        - (M1 * LC1 + M2 * L1) * G * ca.sin(theta[0])
        - M2 * G * L2 * ca.sin(theta[0] + theta[1])
    )
    B = (
        tao[1]
        - M2 * L1 * LC2 * ca.sin(theta[1]) * omega[0] ** 2
        - M2 * G * L2 * ca.sin(theta[0] + theta[1])
    )

    B_mat = np.array([A, B])

    a = I1 + I2 + M2 * L1**2 + 2 * M2 * L1 * LC2 * ca.cos(theta[1])
    b = I2 + M2 * L1 * LC2 * ca.cos(theta[1])
    c = I2 * M2 * L1 * LC2 * ca.cos(theta[1])
    d = I2

    I_matrix_inv_a = (1 / (a * d - b * c)) * d
    I_matrix_inv_b = (1 / (a * d - b * c)) * (-b)
    I_matrix_inv_c = (1 / (a * d - b * c)) * (-c)
    I_matrix_inv_d = (1 / (a * d - b * c)) * a

    I_mat = np.array(
        [[I_matrix_inv_a, I_matrix_inv_b], [I_matrix_inv_c, I_matrix_inv_d]]
    )

    theta_2dot = np.matmul(I_mat, B_mat)

    f_dot = ca.vertcat(omega[0], omega[1], theta_2dot[0], theta_2dot[1])

    return f_dot


# RK4
def rk4step(ode, h, x, u):
    """one step of explicit Runge-Kutta scheme of order four (RK4)

    parameters:
    ode -- odinary differential equations (your system dynamics)
    h -- step of integration
    x -- states
    u -- actions
    """
    k1 = ode(x, u)
    k2 = ode(x + (h / 2) * k1, u)
    k3 = ode(x + (h / 2) * k2, u)
    k4 = ode(x + h * k3, u)

    return x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# Sensor Noise
def sensorNoise(x, mean, sd):
    return x + np.random.normal(mean, sd) * np.diag([1, 1, 0, 0]) @ x


nx = 4
nu = 2
T = 10  # secs simulation
N = 2  # secs horizon

# no of RK4 steps per interval
M = 20
DT = N / M  # the time step for the entire horizon, sampling time for measurements
No_of_control_intervals = int(T / DT)

S_final = np.zeros((nx, No_of_control_intervals + 1))
U_final = np.zeros((nu, No_of_control_intervals))

prev_U = np.zeros((nu, M))
prev_S = np.zeros((nx, M + 1))

S_final[:, 0] = X0

Xk = X0
Xk_np = X0
for s in range(1, No_of_control_intervals + 1):

    # perform an optimization problem for each horizon only take the first control output, hence MPC

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    x_next = ca.Function("x_next", [x, u], [rk4step(dynamics, DT, x, u)])
    meas = ca.Function("meas", [x], [sensorNoise(x, 0, 0.25)])
    opti = ca.Opti()

    u_all = opti.variable(nu, M)
    s_all = opti.variable(nx, M + 1)

    # meaured state after applying control from previous step, with some noise
    meas_state = meas(Xk)

    # Obj function and weighting matrices
    L = 0
    R = np.diag([0.01, 0.01])
    Q = np.diag([1000, 100, 0.1, 0.1])
    for j in range(M):
        if j == 0:
            L += 1 / 2 * (u_all[:, j].T @ R @ u_all[:, j]) + 1 / 2 * (
                meas_state - XN
            ).T @ Q @ (meas_state - XN)
        else:
            L += 1 / 2 * (u_all[:, j].T @ R @ u_all[:, j]) + 1 / 2 * (
                s_all[:, j] - XN
            ).T @ Q @ (s_all[:, j] - XN)
    # final state of horizon, terminal cost
    L += 1 / 2 * (s_all[:, -1] - XN).T @ Q @ (s_all[:, -1] - XN)
    opti.minimize(L)

    # Constraints
    opti.subject_to(s_all[:, 0] - Xk == 0)  # initial condition
    # opti.subject_to( s_all[:,-1] - XN == 0)
    for k in range(M):
        # opti.subject_to(s_all[0:2, k] - THETA_MAX <= 0)
        # opti.subject_to(THETA_MIN - s_all[0:2, k] <= 0)
        opti.subject_to(s_all[2:, k] - OMEGA_MAX <= 0)
        opti.subject_to(OMEGA_MIN - s_all[2:, k] <= 0)
        opti.subject_to(s_all[:, k + 1] - x_next(s_all[:, k], u_all[:, k]) == 0)
        opti.subject_to(u_all[:, k] - U_MAX <= 0)
        opti.subject_to(U_MIN - u_all[:, k] <= 0)

    for n in range(M):
        # first arm
        rad = radius + 0.05
        rad2 = radius2 + 0.05
        x1_in = L1 * np.sin(s_all[0, n])
        y1_in = -L1 * np.cos(s_all[0, n])
        opti.subject_to(
            rad**2 - ((x1_in - center[0]) ** 2 + (y1_in - center[1]) ** 2) <= 0
        )
        opti.subject_to(
            rad2**2 - ((x1_in - center2[0]) ** 2 + (y1_in - center2[1]) ** 2) <= 0
        )

        # points along second arm
        for k in range(1, NUM_OF_POINTS_ALONG_ARM + 1):
            x2_point = x1_in + (L2 * (k / NUM_OF_POINTS_ALONG_ARM)) * np.sin(
                s_all[0, n] + s_all[1, n]
            )
            y2_point = y1_in - (L2 * (k / NUM_OF_POINTS_ALONG_ARM)) * np.cos(
                s_all[0, n] + s_all[1, n]
            )
            d_squared = (x2_point - center[0]) ** 2 + (y2_point - center[1]) ** 2
            d_squared2 = (x2_point - center2[0]) ** 2 + (y2_point - center2[1]) ** 2
            opti.subject_to(rad**2 - d_squared <= 0)
            opti.subject_to(rad2**2 - d_squared2 <= 0)

    # Sol initialization
    opti.set_initial(u_all, prev_U)
    opti.set_initial(s_all, prev_S)

    try:
        opti.solver("ipopt")
        sol = opti.solve()
    except:
        print(opti.debug.value)
        sys.exit(1)

    u_all = sol.value(u_all)
    s_all = sol.value(s_all)

    # shift initialization
    prev_U = np.array(u_all)
    U_last = prev_U[:, -1]
    prev_U[:, 0:-1] = prev_U[:, 1:]
    prev_U[:, -1] = U_last

    prev_S = np.array(s_all)
    S_last = prev_S[:, -1]
    prev_S[:, 0:-1] = prev_S[:, 1:]
    prev_S[:, -1] = S_last

    # extract first control
    U_final[:, s - 1] = np.array(u_all[:, 0])

    # extract second state
    S_final[:, s] = np.array(s_all[:, 1])

    # set next init state i.e second state of current opt problem
    Xk = s_all[:, 1]
    Xk_np = np.array(s_all[:, 1])

# plot the result
fig, ax = plt.subplots()
ax.step(U_final[0, :], "r", label="tao1")
ax.step(U_final[1, :], "y", label="tao2")
plt.xlabel("Time step (s)")
plt.ylabel("Torque (Nm)")
plt.title("Closed Loop MPC Torques (Nearer Objects)")
plt.grid()
plt.legend()

fig, ax = plt.subplots()
ax.plot(S_final[0, :], "b", label="theta1")
ax.plot(S_final[1, :], "k", label="theta2")
ax.plot(S_final[2, :], "g", label="omega1")
ax.plot(S_final[3, :], "m", label="omega2")
plt.xlabel("Time step (s)")
plt.ylabel("Angular Position (rad) and Velocity (rad/s)")
plt.title("Closed Loop MPC States (Nearer Objects)")

plt.grid()
plt.legend()

fig = plt.figure()
ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))
(line,) = ax.plot([], [], lw=2)
(obj,) = ax.plot([], [], lw=2)
(obj2,) = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    obj.set_data([], [])
    obj2.set_data([], [])
    return line, obj, obj2


# arm center coords
centerx = 0
centery = 0


# animation function.  This is called sequentially
def animate(i):

    theta1 = S_final[0, i]
    theta2 = S_final[1, i]

    # first arm end position
    endx1 = np.sin(theta1) * L1
    endy1 = -np.cos(theta1) * L2

    # second arm end position
    endx2 = endx1 + np.sin(theta1 + theta2) * L2
    endy2 = endy1 - np.cos(theta1 + theta2) * L2

    # Circular Object
    theta = np.linspace(0, 2 * np.pi, 150)
    obj_rad = radius
    obj_coords_x = obj_rad * np.sin(theta) + center[0]
    obj_coords_y = obj_rad * np.cos(theta) + center[1]

    obj2_rad = radius2
    obj2_coords_x2 = obj2_rad * np.sin(theta) + center2[0]
    obj2_coords_y2 = obj2_rad * np.cos(theta) + center2[1]

    allXValues = np.concatenate(
        (np.linspace(centerx, endx1, 100), np.linspace(endx1, endx2, 100)), axis=0
    )
    allYvalues = np.concatenate(
        (np.linspace(centery, endy1, 100), np.linspace(endy1, endy2, 100)), axis=0
    )

    line.set_data(allXValues, allYvalues)
    obj.set_data(obj_coords_x, obj_coords_y)
    obj2.set_data(obj2_coords_x2, obj2_coords_y2)

    return line, obj, obj2


anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=No_of_control_intervals + 1,
    interval=125,
    blit=True,
)

os.makedirs("results", exist_ok=True)

anim.save(filename="results/closedLoop.gif", writer="pillow")

num_key_frames = 8

with Image.open("results/closedLoop.gif") as im:
    for i in range(num_key_frames):
        im.seek(im.n_frames // num_key_frames * i)
        im.save("results/closedLoop_{}.png".format(i))

plt.show()
