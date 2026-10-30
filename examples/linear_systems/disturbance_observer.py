"""
Python script one_d_disturbance_observer.py demonstrating a disturbance observer.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy
"""

# %%[markdown]
# # 1D Mobile Robot Disturbance Observer with Controller
# Copyright &copy; 2021-2026 by
# [Joshua A. Marshall](https://www.queensu.ca/offroad-robotics/people/joshua-marshall/)

# %%[markdown]
# ## Simulation Setup

from IPython.display import display
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import Cart
from mobotpy import BOOK_STYLE

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 20.0
T = 0.1

# Set the side length of the mobile robot [m]
LENGTH = 1.0

# Set some variables that describe the desired behaviour
ZETA = 1.1
OMEGA_N = np.sqrt(0.3)

# Define the mobile robot mass [kg]
M = 10.0

# Slope of the terrain [radians]
SLOPE = np.pi / 30

# Compute the controller pole locations
LAMBDA_SC = np.roots([1, 2 * ZETA * OMEGA_N, OMEGA_N**2])
LAMBDA_ZC = np.exp(LAMBDA_SC * T)

# Function that models the mobile robot and sensor(s) in discrete time
F = np.array([[1, T], [0, 1]])
G = np.array([[T**2 / (2 * M)], [T / M]])
H = np.array([[1, 0]])

# Find gain matrix K that places the poles at lambda_zc
K = signal.place_poles(F, G, LAMBDA_ZC)

# Function that models the mobile robot plus disturbance in discrete time
F_A = np.vstack((np.hstack((F, G)), [[0, 0, 1]]))
G_A = np.vstack((G, [[0]]))
H_A = np.hstack((H, [[0]]))

# Choose estimator gains for stability
LAMBDA_SO = 2.0 * LAMBDA_SC
LAMBDA_ZO = np.hstack((np.exp(LAMBDA_SO * T), 0.1))
LT = signal.place_poles(F_A.T, H_A.T, LAMBDA_ZO).gain_matrix.T

# %%[markdown]
# ## Function Definitions


def robot(x_k, u_k, w_k=0.0):
    """Discrete-time 1D dynamic mobile robot model."""
    x_new = F @ x_k + G @ [u_k] + G @ [w_k]
    return x_new


def controller(x_k):
    """Full state feedback controller."""
    u_k = -K.gain_matrix[0] @ x_k
    return u_k


def observer(z_hat_k, u_k, y_k):
    """Observer-based state and disturbance estimator."""
    z_hat_new = F_A @ z_hat_k + G_A @ [u_k] + LT @ ([y_k] - H_A @ z_hat_k)
    return z_hat_new


def simulation():
    """Run the 1D disturbance observer simulation."""

    # Create an array of time values [s]
    t = np.arange(0, SIM_TIME, T)
    time_steps = np.size(t)

    # Initialize arrays that will be populated with our inputs and augmented state
    x = np.zeros((2, time_steps))
    u = np.zeros(time_steps)
    x_hat = np.zeros((2, time_steps))
    w_hat = np.zeros(time_steps)
    z_hat = np.zeros((3, time_steps))

    # Set the initial position [m], velocity [m/s], and force input [N]
    x[0, 0] = 1.0
    x[1, 0] = 0.0
    u[0] = 0.0

    # Set the initial estimated state (different from the actual state)
    x_hat[0, 0] = 0.0
    x_hat[1, 0] = 0.0
    w_hat[0] = 0.0
    z_hat[:, 0] = np.hstack((x_hat[:, 0], w_hat[0]))

    # Run the simulation for time step k
    for k in range(1, time_steps):
        y = x[0, k - 1]
        z_hat[:, k] = observer(z_hat[:, k - 1], u[k - 1], y)
        x_hat[:, k] = z_hat[0:2, k]
        w_hat[k] = z_hat[2, k]
        x[:, k] = robot(x[:, k - 1], u[k - 1], -M * 9.81 * np.sin(SLOPE))
        u[k] = controller(x_hat[:, k]) + w_hat[k]

    return t, x, u, x_hat, w_hat


def plotting(t, x, u, x_hat, w_hat):
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
        pdf_path = "../mobotpy-book/figs/chap4/"
        gif_path = "../mobotpy-book/gifs/chap4/"
    else:
        pdf_path = ""
        gif_path = ""

    # Plot the states (x) and input (u) vs time (t)
    plt.figure(1)
    ax1a = plt.subplot(411)
    plt.plot(t, x[0, :], "C0", label="Actual")
    plt.step(t, x_hat[0, :], "C2--", where="post", label="Estimated")
    plt.ylabel(r"$x_1$ [m]")
    plt.setp(ax1a, xticklabels=[])
    plt.legend(loc="lower right")
    ax1b = plt.subplot(412)
    plt.plot(t, x[1, :], "C0")
    plt.step(t, x_hat[1, :], "C2--", where="post")
    plt.ylabel(r"$x_2$ [m/s]")
    plt.setp(ax1b, xticklabels=[])
    ax1c = plt.subplot(413)
    plt.plot(t, -9.81 * M * np.sin(SLOPE) * np.ones(np.size(t)), "C0")
    plt.step(t, w_hat, "C2--", where="post")
    plt.ylabel(r"$w$ [N]")
    plt.setp(ax1c, xticklabels=[])
    plt.subplot(414)
    plt.step(t, u, "C1", where="post")
    plt.ylabel(r"$u$ [N]")
    plt.xlabel(r"$t$ [s]")

    if export_plots:
        # Save the figure to a PDF file
        plt.savefig(pdf_path + "one_d_disturbance_observer_fig1.pdf")
    else:
        # Show the plot to the screen
        plt.show()

    # Let's use the Cart class to create an animation
    cart = Cart(LENGTH)

    # Create and save the animation
    ani = cart.animate(
        x[0, :], T, export_plots, gif_path + "one_d_disturbance_observer.gif"
    )

    if not export_plots:
        # Show animation in HTML output if you are using IPython or Jupyter notebooks
        plt.rc("animation", html="jshtml")
        display(ani)
        plt.close()


def main():
    """Run the script."""

    # Run the simulation
    t, x, u, x_hat, w_hat = simulation()

    # Plot the results and create an animation
    plotting(t, x, u, x_hat, w_hat)


# %%[markdown]
# ## Run Simulation

if __name__ == "__main__":
    main()
