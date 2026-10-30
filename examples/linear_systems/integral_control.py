"""
Python script one_d_integral_control.py demonstrating integral control.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy
"""

# %%[markdown]
# # 1D Mobile Robot Control in with Integral Action
# Copyright &copy; 2021-2026 by
# [Joshua A. Marshall](https://www.queensu.ca/offroad-robotics/people/joshua-marshall/)

# %%[markdown]
# ## Simulation Setup

# %%
# Import the required libraries
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

# Set some parameters that describe the desired behaviour
ZETA = 1.1
OMEGA_N = np.sqrt(0.3)

# Compute the pole locations
LAMBDA_S = np.roots([1, 2 * ZETA * OMEGA_N, OMEGA_N**2])
LAMBDA_Z = np.hstack((np.exp(LAMBDA_S * T), 0.5))

# Define the vehicle mass [kg]
M = 10.0

# Define the system matrices
F = np.array([[1, T], [0, 1]])
G = np.array([[T**2 / (2 * M)], [T / M]])
H = np.array([[-1, 0]])

# Augmented state system matrices
A = np.hstack((np.vstack((F, H)), np.array([[0], [0], [1]])))
B = np.vstack((G, 0))

# Find gain matrix K that places the poles inside the unit disk
K = signal.place_poles(A, B, LAMBDA_Z)

# %%[markdown]
# ## Function Definitions


def robot(x_k, u_k):
    """Discrete-time 1D mobile robot model on a slope."""
    x_new = F @ x_k + G @ [u_k] - G @ [M * 9.81 * np.sin(np.pi / 30)]
    return x_new


def integrator(x_k, xi_k):
    """Augmented state integrator."""
    xi_new = xi_k - x_k[0]
    return xi_new


def controller(x_k, xi_k):
    """State feedback controller with integral action."""
    u_new = -K.gain_matrix @ np.hstack((x_k, xi_k))
    return u_new[0]


def simulation():
    """Run the 1D mobile robot simulation with integral control."""

    # Create an array of time values [s]
    t = np.arange(0, SIM_TIME, T)
    time_steps = np.size(t)

    # Initialize arrays to be populated with inputs and states
    x = np.zeros((2, time_steps))
    xi = np.zeros(time_steps)
    u = np.zeros(time_steps)

    # Set the initial position [m], velocity [m/s], and force input [N]
    x[0, 0] = 1.0
    x[1, 0] = 0.0
    u[0] = 0.0

    # Run the simulation
    for k in range(0, time_steps - 1):
        x[:, k + 1] = robot(x[:, k], u[k])
        xi[k + 1] = integrator(x[:, k], xi[k])
        u[k + 1] = controller(x[:, k + 1], xi[k + 1])

    return t, x, u


def plotting(t, x, u):
    """Plot the results of the simulation and create an animation."""

    # Set `use_tex` to `True` to use an external LaTeX installation in the plots
    use_tex = False

    # Set `export_plots` to `True` to save the plots and animations to files
    export_plots = False

    # Set `book_format` to `True` to use fancy LaTeX fonts (requires `use_tex = True`)
    book_format = False

    # Use LaTeX formatting in the plots
    if use_tex:
        plt.rc("text", usetex=True)
        if book_format:
            plt.style.use(BOOK_STYLE)

    # Set (change) the path to save the plots and animations
    if export_plots:
        pdf_path = "../mobotpy-book/figs/chap3/"
        gif_path = "../mobotpy-book/gifs/chap3/"
    else:
        pdf_path = ""
        gif_path = ""

    # Plot the state `x` and input `u` vs time `t`
    plt.figure(1)
    ax1a = plt.subplot(311)
    plt.plot(t, x[0, :], "C0")
    plt.ylabel(r"$x_1$ [m]")
    plt.setp(ax1a, xticklabels=[])
    ax1b = plt.subplot(312)
    plt.plot(t, x[1, :], "C0")
    plt.ylabel(r"$x_2$ [m/s]")
    plt.setp(ax1b, xticklabels=[])
    plt.subplot(313)
    plt.step(t, u, "C1", where="post")
    plt.ylabel(r"$u$ [N]")
    plt.xlabel(r"$t$ [s]")

    if export_plots is True:
        # Save the figure to a PDF file
        plt.savefig(pdf_path + "one_d_integral_control_fig1.pdf")
    else:
        # Show the plot to the screen
        plt.show()

    # Let's use the Cart class to create an animation
    vehicle = Cart(LENGTH)

    # Create and save the animation
    ani = vehicle.animate(
        x[0, :], T, export_plots, gif_path + "one_d_integral_control.gif"
    )

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
