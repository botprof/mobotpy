"""
Python script one_d_dynamic.py for simulating a linear 1D dynamic mobile robot.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy
"""

# %%[markdown]
# # 1D Mobile Robot Dynamics
# Copyright &copy; 2021-2026 by
# [Joshua A. Marshall](https://www.queensu.ca/offroad-robotics/people/joshua-marshall/)

# %%[markdown]
# ## Simulation Setup

# Import the required libraries
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from mobotpy.models import Cart
from mobotpy import BOOK_STYLE

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.04

# Set the side length of the mobile robot [m]
LENGTH = 1.0

# Set the mass of the mobile robot [kg]
M = 10.0

# Define the mobile robot model matrices
F = np.array([[1, T], [0, 1]])
G = np.array([[T**2 / (2 * M)], [T / M]])

# %%[markdown]
# ## Function Definitions


def robot(x_k, u_k):
    """Discrete-time 1D dynamic mobile robot model."""
    x_new = F @ x_k + G @ [u_k]
    return x_new


def simulation():
    """Run the 1D dynamic mobile robot simulation."""

    # Create an array of time values [s]
    t = np.arange(0, SIM_TIME, T)
    time_steps = np.size(t)

    # Initialize the state and input arrays
    x = np.zeros((2, time_steps))
    u = np.zeros(time_steps)

    # Set the initial position [m], velocity [m/s], and force input [N]
    x[0, 0] = 0.0
    x[1, 0] = 0.0
    u[0] = 0.0

    # Run the simulation
    for k in range(0, time_steps - 1):
        x[:, k + 1] = robot(x[:, k], u[k])
        u[k + 1] = np.sin((k + 1) * T)

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
        pdf_path = "../mobotpy-book/figs/chap2/"
        gif_path = "../mobotpy-book/gifs/chap2/"
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

    if export_plots:
        # Save the figure to a PDF file
        plt.savefig(pdf_path + "one_d_dynamic_fig1.pdf")
    else:
        # Show the plot to the screen
        plt.show()

    # Let's use the Cart class to create an animation
    cart = Cart(LENGTH)

    # Create and save the animation
    ani = cart.animate(x[0, :], T, export_plots, gif_path + "one_d_dynamic.gif")

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
