"""
Script one_d_linear_mpc.py demonstrating linear MPC.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from numpy.linalg import matrix_power
from numpy.linalg import inv
from mobotpy import BOOK_STYLE

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 15.0
T = 0.1

# %%[markdown]
# ## Simulation Setup

# Vehicle mass [kg]
M = 1.0

# Discrete time vehicle model
F = np.array([[1, T], [0, 1]])
G = np.array([[T**2 / (2 * M)], [T / M]])
n = G.shape[0]

# Lookahead time steps
P = 100

# Decide on state and input cost matrices
SMALL_Q = np.diag([2.0, 1.0])
SMALL_R = np.diag([0.1])
Q = np.kron(np.eye(P), SMALL_Q)
R = SMALL_R * np.eye(P)

# %%[markdown]
# ## Function Definitions


def mpc_gain():
    """Compute the MPC gain matrix."""

    # Construct the matrices m_mat and l_mat
    l_mat = np.zeros((n * P, n))
    m_mat = np.zeros((n * P, P))
    for i in range(0, P):
        l_mat[n * i : n * i + n, 0:n] = matrix_power(F, i + 1)
        for j in range(0, P - i):
            m_mat[n * (P - i) - n : n * (P - i), j : j + 1] = (
                matrix_power(F, P - i - j - 1) @ G
            )

    # Compute the MPC gain matrix
    k_mpc = inv(m_mat.T @ Q @ m_mat + R) @ m_mat.T @ Q

    return k_mpc, l_mat


def lqr_gain():
    """Compute the LQR gain matrix."""

    # Solve the discrete-time algebraic Riccati equation (DARE)
    p_mat = solve_discrete_are(F, G, SMALL_Q, SMALL_R)

    # Compute the LQR gain matrix
    k_opt = np.linalg.inv(G.T @ p_mat @ G + SMALL_R) @ (G.T @ p_mat @ F)

    return k_opt


def simulation():
    """Run the 1D kinematic mobile robot simulation."""

    # Create an array of time values [s]
    t = np.arange(0, SIM_TIME, T)
    time_steps = np.size(t)

    # Set the desired trajectory to take a step mid way through the trajectory
    x_ref = np.zeros((2, time_steps + P))
    for k in range(int(time_steps / 2), time_steps + P):
        x_ref[0, k] = 10
        x_ref[1, k] = 0

    # Set up some more arrays for MPC and LQR
    x_mpc = np.zeros((n, time_steps))
    u_mpc = np.zeros((1, time_steps))
    xi_ref = np.zeros(n * P)
    x_lqr = np.zeros((n, time_steps))
    u_lqr = np.zeros((1, time_steps))

    # Set the initial conditions
    x_mpc[0, 0] = 3
    x_mpc[1, 0] = 0
    x_lqr[0, 0] = x_mpc[0, 0]
    x_lqr[1, 0] = x_mpc[1, 0]

    # Compute the MPC and LQR gain matrices
    k_mpc, l_mat = mpc_gain()
    k_lqr = lqr_gain()

    # Simulate the the closed-loop system with MPC
    for k in range(1, time_steps):
        x_mpc[:, k] = F @ x_mpc[:, k - 1] + G @ u_mpc[:, k - 1]
        for j in range(0, P):
            xi_ref[n * j : n * j + n] = x_ref[:, k + j]
        u = k_mpc @ (xi_ref - l_mat @ x_mpc[:, k - 1])
        u_mpc[:, k] = u[0]

    # Simulate the closed-loop system with LQR
    for k in range(1, time_steps):
        x_lqr[:, k] = F @ x_lqr[:, k - 1] + G @ u_lqr[:, k - 1]
        u_lqr[:, k] = k_lqr @ (x_ref[:, k - 1] - x_lqr[:, k - 1])

    return t, x_ref, x_mpc, x_lqr, u_mpc, u_lqr


def plotting(t, x_ref, x_mpc, x_lqr, u_mpc, u_lqr):
    """Plot the results of the simulation and create an animation."""

    # Set `use_tex` to `True` to use an external LaTeX installation in the plots
    use_tex = True

    # Set `export_plots` to `True` to save the plots and animations to files
    export_plots = True

    # Set `book_format` to `True` to use fancy LaTeX fonts (requires `use_tex = True`)
    book_format = True

    # Use LaTeX formatting in the plots
    if use_tex:
        plt.rc("text", usetex=True)
        if book_format:
            plt.style.use(BOOK_STYLE)

    # Set (change) the path to save the plots and animations
    if export_plots:
        pdf_path = "../mobotpy-book/figs/chap5/"
    else:
        pdf_path = ""

    # Find the number of time steps
    time_steps = np.size(t)

    plt.figure(1)
    plt.subplot(311)
    plt.plot(t, x_ref[0, 0:time_steps], "C2--")
    plt.plot(t, x_mpc[0, :], "C0")
    plt.plot(t, x_lqr[0, :], "C1")
    plt.grid(color="0.95")
    plt.ylabel(r"$x_1$ [m]")
    plt.legend(["Desired", "MPC", "LQR"], loc="lower right")
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.subplot(312)
    plt.plot(t, x_ref[1, 0:time_steps], "C2--")
    plt.plot(t, x_mpc[1, :], "C0")
    plt.plot(t, x_lqr[1, :], "C1")
    plt.grid(color="0.95")
    plt.ylabel(r"$x_2$ [m/s]")
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.subplot(313)
    plt.plot(t, u_mpc[0, :], "C0")
    plt.plot(t, u_lqr[0, :], "C1")
    plt.grid(color="0.95")
    plt.ylabel(r"$u$ [N]")
    plt.xlabel(r"$t$ [s]")

    if export_plots:
        # Save the figure to a PDF file
        plt.savefig(pdf_path + "one_d_linear_mpc_fig1.pdf")
    else:
        # Show the plot to the screen
        plt.show()


def main():
    """Run the script."""

    # Run the simulation
    t, x_ref, x_mpc, x_lqr, u_mpc, u_lqr = simulation()

    # Plot the results and create an animation
    plotting(t, x_ref, x_mpc, x_lqr, u_mpc, u_lqr)


# %%[markdown]
# ## Run Simulation

if __name__ == "__main__":
    main()
