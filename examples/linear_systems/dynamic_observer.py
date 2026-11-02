"""
Python script dynamic_observer.py Luenberger observer example.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy
"""

# %%[markdown]
# # 1D Mobile Robot Luenberger Observer
# Copyright &copy; 2021-2026 by
# [Joshua A. Marshall](https://www.queensu.ca/offroad-robotics/people/joshua-marshall/)

# %%[markdown]
# ## Simulation Setup

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from mobotpy import BOOK_STYLE

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 5.0
T = 0.1

# Set the mass of the mobile robot [kg]
M = 10.0

# Function that models the mobile robot and sensor(s) in discrete time
F = np.array([[1, T], [0, 1]])
G = np.array([[T**2 / (2 * M)], [T / M]])
H = np.array([[1, 0]])

# Choose estimator gains for stability
LAMBDA_Z = np.array([0.5, 0.4])
L = signal.place_poles(F.T, H.T, LAMBDA_Z).gain_matrix.T

# %%[markdown]
# ## Function Definitions


def robot(x_k, u_k):
    """Discrete-time 1D dynamic mobile robot model."""
    x_new = F @ x_k + G @ [u_k]
    return x_new


def observer(x_hat_k, u_k, y_k):
    """Observer-based state estimator."""
    x_hat_new = F @ x_hat_k + G @ [u_k] + L @ ([y_k] - H @ x_hat_k)
    return x_hat_new


def simulation():
    """Run the 1D dynamic observer simulation."""

    # Create an array of time values [s]
    t = np.arange(0, SIM_TIME, T)
    time_steps = np.size(t)

    # Initialize arrays that will be populated with our inputs and states
    x = np.zeros((2, time_steps))
    u = np.zeros(time_steps)
    x_hat = np.zeros((2, time_steps))

    # Set the initial position [m], velocity [m/s], and force input [N]
    x[0, 0] = 1.0
    x[1, 0] = 0.0
    u[0] = 0.0

    # Set the initial estimated state (different from the actual state)
    x_hat[0, 0] = 0.0
    x_hat[1, 0] = 0.0

    # Run the simulation for time step k
    for k in range(0, time_steps - 1):
        y = x[0, k]
        x_hat[:, k + 1] = observer(x_hat[:, k], u[k], y)
        x[:, k + 1] = robot(x[:, k], u[k])
        u[k + 1] = 2.0 * np.sin((k + 1) * T)

    return t, x, x_hat


def plotting(t, x, x_hat):
    """Plot the results of the simulation."""

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
        pdf_path = "../mobotpy-book/figs/chap4/"
    else:
        pdf_path = ""

    # Plot the state `x` and input `u` vs time `t`
    plt.figure(1)
    ax1a = plt.subplot(211)
    plt.plot(t, x[0, :], "C0")
    plt.step(t, x_hat[0, :], "C2--", where="post")
    plt.ylabel(r"$x_1$ [m]")
    plt.setp(ax1a, xticklabels=[])
    plt.legend(["Actual", "Estimated"])
    plt.subplot(212)
    plt.plot(t, x[1, :], "C0")
    plt.step(t, x_hat[1, :], "C2--", where="post")
    plt.ylabel(r"$x_2$ [m/s]")
    plt.xlabel(r"$t$ [s]")

    if export_plots:
        # Save the figure to a PDF file
        plt.savefig(pdf_path + "dynamic_observer_fig1.pdf")
    else:
        # Show the plot to the screen
        plt.show()


def main():
    """Run the script."""

    # Run the simulation
    t, x, x_hat = simulation()

    # Plot the results and create an animation
    plotting(t, x, x_hat)


# %%[markdown]
# ## Run Simulation

if __name__ == "__main__":
    main()
