import os

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Model Params
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

# Objects
center = (1.75, 1.75)
radius = 0.5

center2 = (1.75, -1.75)
radius2 = 0.5
NUM_OF_POINTS_ALONG_ARM = 5

# Sim Params
T = 10
h = 0.1
N = int(T / h)
end_state_time = int(N / 10)


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


# OCP formulation
nu = 2
ns = 4

X_ca = np.zeros((X0.size, N))
X_ca[:, 0] = X0

x = ca.MX.sym("x", 4)
u = ca.MX.sym("u", 2)

x_next = ca.Function("x_next", [x, u], [rk4step(dynamics, h, x, u)])
opti = ca.Opti()

u_all = opti.variable(nu, N)
s_all = opti.variable(ns, N + 1)

L = 0
for i in range(N):
    L += 1 / 2 * (np.matmul(u_all[:, i].T, u_all[:, i])) + 1 / 2 * (
        np.matmul((s_all[:, i] - XN).T, (s_all[:, i] - XN))
    )
L += 1 / 2 * (np.matmul((s_all[:, -1] - XN).T, (s_all[:, -1] - XN)))

opti.minimize(L)

opti.subject_to(s_all[:, 0] - X0 == 0)  # initial condition
for i in range(N):
    opti.subject_to(s_all[2:, i] - OMEGA_MAX <= 0)
    opti.subject_to(OMEGA_MIN - s_all[2:, i] <= 0)
    opti.subject_to(s_all[:, i + 1] - x_next(s_all[:, i], u_all[:, i]) == 0)
    opti.subject_to(u_all[:, i] - U_MAX <= 0)
    opti.subject_to(U_MIN - u_all[:, i] <= 0)

# Final Constraint
for j in range(end_state_time):
    opti.subject_to(s_all[:, N - j] - XN == 0)

# Object constraints
# assume points along the bars at equal intervals must not be within the circular object
# s_all := [theta1, theta2, omega1, omega2]
# use theta1 and theta2 to calculate co-ordinates
# formulate as inequality constraints

for i in range(N):
    # first arm
    rad = radius
    rad2 = radius2
    x1_in = L1 * np.sin(s_all[0, i])
    y1_in = -L1 * np.cos(s_all[0, i])
    opti.subject_to(
        rad * 2 - ((x1_in - center[0]) ** 2 + (y1_in - center[1]) ** 2) <= 0
    )
    opti.subject_to(
        rad2 * 2 - ((x1_in - center2[0]) ** 2 + (y1_in - center2[1]) ** 2) <= 0
    )

    # points along second arm
    for k in range(1, NUM_OF_POINTS_ALONG_ARM + 1):
        x2_point = x1_in + (L2 * (k / NUM_OF_POINTS_ALONG_ARM)) * np.sin(
            s_all[0, i] + s_all[1, i]
        )
        y2_point = y1_in - (L2 * (k / NUM_OF_POINTS_ALONG_ARM)) * np.cos(
            s_all[0, i] + s_all[1, i]
        )
        d_squared = (x2_point - center[0]) ** 2 + (y2_point - center[1]) ** 2
        d_squared2 = (x2_point - center2[0]) ** 2 + (y2_point - center2[1]) ** 2
        opti.subject_to(rad**2 - d_squared <= 0)
        opti.subject_to(rad2**2 - d_squared2 <= 0)
opti.solver("ipopt")
sol = opti.solve()

u_sol = sol.value(u_all)
s_sol = sol.value(s_all)
# s_sol = np.insert(s_sol, 0, X0)

# plot the result
fig, ax = plt.subplots()
ax.step(u_sol[0, :], "r", label="tao1")
ax.step(u_sol[1, :], "y", label="tao2")
plt.xlabel("Time step (s)")
plt.ylabel("Torque (Nm)")
plt.title("Open Loop Approach Controls (Further Objects)")

fig, ax = plt.subplots()
ax.plot(s_sol[0, :], "b", label="theta1")
ax.plot(s_sol[1, :], "k", label="theta2")
ax.plot(s_sol[2, :], "g", label="omega1")
ax.plot(s_sol[3, :], "m", label="omega2")
plt.xlabel("Time step (s)")
plt.ylabel("Angular Position (rad) and Velocity (rad/s)")
plt.title("Open Loop Approach States (Further Objects)")

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

    global X_ca

    theta1 = s_sol[0, i]
    theta2 = s_sol[1, i]

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
    fig, animate, init_func=init, frames=N, interval=50, blit=True
)

os.makedirs("results", exist_ok=True)

anim.save(filename="results/openLoop.gif", writer="Pillow")

plt.show()
