"""
Python script kinematic_control.py for kinematic proportional control problem.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy
"""

# %%[markdown]
# # Simple Proportional Control Example
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
SIM_TIME = 10.0
T = 0.4

# Set the side length of the mobile robot [m]
LENGTH = 1.0

# Set the desired position [m] and controller gain
X_REF = 4.0
K_P = 1.0


# %%[markdown]
# ## Function Definitions


def robot(x_k, u_k):
    """Discrete-time 1D kinematic mobile robot model."""
    x_new = x_k + T * u_k
    return x_new


def simulation():
    """Run the 1D kinematic mobile robot simulation."""

    # Create an array of time values [s]
    t = np.arange(0, SIM_TIME, T)
    time_steps = np.size(t)

    # Initialize the state and input arrays
    x = np.zeros(time_steps)
    u = np.zeros(time_steps)

    # Set the initial position [m] and initial input [m/s]
    x[0] = 1.0
    u[0] = 0.0

    # Run the simulation
    for k in range(0, time_steps - 1):
        x[k + 1] = robot(x[k], u[k])
        u[k + 1] = K_P * (X_REF - x[k + 1])

    return t, x, u


def plotting(t, x, u):
    """Plot the results and create an animation."""

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
        pdf_path = "../mobotpy-book/figs/chap3/"
        gif_path = "../mobotpy-book/gifs/chap3/"
    else:
        pdf_path = ""
        gif_path = ""

    # Plot the state `x` and input `u` vs time `t`
    plt.figure(1)
    ax1a = plt.subplot(211)
    plt.plot(t, x, "C0")
    plt.ylabel(r"$x$ [m]")
    plt.setp(ax1a, xticklabels=[])
    plt.subplot(212)
    plt.step(t, u, "C1", where="post")
    plt.ylabel(r"$u$ [m/s]")
    plt.xlabel(r"$t$ [s]")

    if export_plots:
        # Save the figure to a PDF file
        plt.savefig(pdf_path + "kinematic_control_fig1.pdf")
    else:
        # Show the plot to the screen
        plt.show()

    # Let's use the Cart class to create an animation
    cart = Cart(LENGTH)

    # Create and save the animation
    ani = cart.animate(x, T, export_plots, gif_path + "kinematic_control.gif")

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
