"""
Script discrete_lqr.py for control of a 1D mobile robot with LQR.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy
"""

# %%[markdown]
# # 1D Mobile Robot Control in Discrete Time with LQR
# Copyright &copy; 2021-2026 by
# [Joshua A. Marshall](https://www.queensu.ca/offroad-robotics/people/joshua-marshall/)

# %%[markdown]
# ## Simulation Setup

from IPython.display import display
from scipy.linalg import solve_discrete_are
import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import Cart
from mobotpy import BOOK_STYLE

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 25.0
T = 0.5

# Set the side length of the vehicle [m]
LENGTH = 1.0

# Define the vehicle mass [kg]
M = 10.0

# Define the system matrices
F = np.array([[1, T], [0, 1]])
G = np.array([[T**2 / (2 * M)], [T / M]])

# Set the weights for the LQR cost function
Q = np.diag([10, 0.1])
R = np.zeros(3)
R[0] = 0.1
R[1] = 1.0
R[2] = 10.0

# %%[markdown]
# ## Function Definitions


def robot(x_k, u_k):
    """Discrete-time 1D dynamic vehicle model."""
    x_new = F @ x_k + G @ [u_k]
    return x_new


def controller(x_k, k_matrix):
    """Proportional controller."""
    u_new = -k_matrix @ x_k
    return u_new[0]


def lqr_gain(q_mat, r_mat):
    """Compute the LQR gain matrix."""

    # Solve the discrete-time algebraic Riccati equation (DARE)
    p_mat = solve_discrete_are(F, G, q_mat, r_mat)

    # Compute the LQR gain matrix
    k_opt = np.linalg.inv(G.T @ p_mat @ G + r_mat) @ (G.T @ p_mat @ F)

    return k_opt


def simulation():
    """Run the 1D dynamic mobile robot simulation."""

    # Create an array of time values [s]
    t = np.arange(0, SIM_TIME, T)
    time_steps = np.size(t)

    # Initialize arrays that will be populated with our inputs and states
    x = np.zeros((2, time_steps, 3))
    u = np.zeros((time_steps, 3))

    # Set the initial position [m], velocity [m/s], and force input [N]
    x[0, 0, :] = 1.0
    x[1, 0, :] = 0.0
    u[0, :] = 0.0

    # Run the simulation for each weight R
    for j in range(0, 3):

        # Compute the LQR gain matrix
        k_opt = lqr_gain(Q, R[j])

        # Run the simulation
        for k in range(0, time_steps - 1):
            x[:, k + 1, j] = robot(x[:, k, j], u[k, j])
            u[k + 1, j] = controller(x[:, k + 1, j], k_opt)

    return t, x, u


def plotting(t, x, u):
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
        gif_path = "../mobotpy-book/gifs/chap5/"
    else:
        pdf_path = ""
        gif_path = ""

    # Define a list of line styles
    line_styles = ["-", "--", "-.", ":"]

    # Plot the state `x` and input `u` vs time `t`
    plt.figure(1)
    ax1a = plt.subplot(311)
    for j in range(0, 3):
        plt.plot(
            t, x[0, :, j], "C" + str(j), linestyle=line_styles[j % len(line_styles)]
        )
    plt.grid(color="0.95")
    if use_tex is True:
        plt.legend(
            [r"$\bm{R}=0.10$", r"$\bm{R}=1.0$", r"$\bm{R}=10$"],
            loc="upper right",
        )
    else:
        plt.legend(
            [r"$\mathbf{R}=0.10$", r"$\mathbf{R}=1.0$", r"$\mathbf{R}=10$"],
            loc="upper right",
        )
    plt.ylabel(r"$x_1$ [m]")
    plt.setp(ax1a, xticklabels=[])
    ax1b = plt.subplot(312)
    for j in range(0, 3):
        plt.plot(
            t, x[1, :, j], "C" + str(j), linestyle=line_styles[j % len(line_styles)]
        )
    plt.grid(color="0.95")
    plt.ylabel(r"$x_2$ [m/s]")
    plt.setp(ax1b, xticklabels=[])
    plt.subplot(313)
    for j in range(0, 3):
        plt.step(
            t,
            u[:, j],
            "C" + str(j),
            where="post",
            linestyle=line_styles[j % len(line_styles)],
        )
    plt.grid(color="0.95")
    plt.ylabel(r"$u$ [N]")
    plt.xlabel(r"$t$ [s]")

    if export_plots is True:
        # Save the figure to a PDF file
        plt.savefig(pdf_path + "discrete_lqr_fig1.pdf")
    else:
        # Show the plot to the screen
        plt.show()

    # Let's use the Cart class to create an animation
    cart = Cart(LENGTH)

    # Create and save the animation
    ani = cart.animate(x[0, :, 0], T, export_plots, gif_path + "discrete_lqr.gif")

    if not export_plots:
        # Show animation in HTML output if you are using IPython or Jupyter notebooks
        plt.rc("animation", html="jshtml")
        display(ani)
        plt.close()


def main():
    """Run the script."""

    # Run the simulation
    t, x, u = simulation()

    # Plot the results and create an animation
    plotting(t, x, u)


# %%[markdown]
# ## Run Simulation

if __name__ == "__main__":
    main()
