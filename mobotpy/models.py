"""
Python module models.py for various mobile robot models.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy

This module contains classes for various mobile robot models, including:
    * [Cart] 1D mobile robot class (i.e., a simple cart)
    * [DiffDrive] Differential drive vehicle class
    * [Tricycle] Tricycle or planar bicycle class
    * [Ackermann] Ackermann steered class
    * [FourWheelSteering] Four-wheel steering class
"""

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches
from mobotpy import graphics


class Cart:
    """1D vehicle class (i.e., a simple cart).

    Parameters
    ----------
    length : float
        Length of the cart [m].
    """

    def __init__(self, length):
        """Constructor method."""
        self.d = length

    def draw(self, x):
        """Finds the points to draw simple rectangular cart.

        The cart has position x and length d.  The resulting cart has a height
        that is half the length.
        """
        x_points = np.array(5)
        y_points = np.array(5)
        x_points = [
            x - self.d / 2,
            x - self.d / 2,
            x + self.d / 2,
            x + self.d / 2,
            x - self.d / 2,
        ]
        y_points = [-self.d / 4, self.d / 4, self.d / 4, -self.d / 4, -self.d / 4]
        return x_points, y_points

    def animate(self, x, dt, save_ani=False, filename="animate_cart.gif"):
        """Create an animation of a simple 1D cart.

        Returns animation object for array of 1D cart positions x with time
        increments dt [s], cart width d [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default filename is 'animate_cart.gif').
        """
        fig, ax = plt.subplots()
        plt.plot([np.min(x) - self.d, np.max(x) + self.d], [0, 0], "k--")
        plt.xlabel(r"$x$ [m]")
        ax.set_xlim([np.min(x) - self.d, np.max(x) + self.d])
        plt.yticks([])
        plt.axis("equal")
        (polygon,) = ax.fill([], [], "C0", alpha=0.5)
        (line,) = plt.plot([], [], "ko")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        # Initialization function

        def init():
            polygon.set_xy(np.empty([5, 2]))
            line.set_data([], [])
            time_text.set_text("")
            return polygon, line, time_text

        # Function to draw cart

        def movie(k):
            x_coords, y_coords = self.draw(x[k])
            a = [x_coords, y_coords]
            polygon.set_xy(np.transpose(a))
            line.set_data([x[k]], [0])
            time_text.set_text(rf"$t$ = {k * dt:.1f} s")
            return polygon, line, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x), max(1, int(1 / dt / 10))),
            init_func=init,
            interval=dt * 1000,
            blit=True,
            repeat=False,
        )
        # Save to a file if requested
        if save_ani:
            ani.save(filename, fps=min(1 / dt, 10))
        # Return the figure object
        return ani


class DiffDrive:
    """Differential-drive vehicle class.

    Parameters
    ----------
    ell : float
        The track length of the vehicle [m].
    """

    def __init__(self, ell):
        """Constructor method."""
        self.ell = ell

    def f(self, x, u):
        """Differential drive kinematic vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 3
            The vehicle's state (x, y, theta).
        u : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).

        Returns
        -------
        f : ndarray of length 3
            The rate of change of the vehicle states.
        """
        f = np.zeros(3)
        f[0] = 0.5 * (u[0] + u[1]) * np.cos(x[2])
        f[1] = 0.5 * (u[0] + u[1]) * np.sin(x[2])
        f[2] = 1.0 / self.ell * (u[1] - u[0])
        return f

    def uni2diff(self, u_in):
        """
        Convert speed and angular rate inputs to differential drive wheel speeds.

        Parameters
        ----------
        u_in : ndarray of length 2
            The speed and turning rate of the vehicle (v, omega).

        Returns
        -------
        u_out : ndarray of length 2
            The left and right wheel speeds (v_l, v_r).
        """
        v = u_in[0]
        omega = u_in[1]
        v_l = v - self.ell / 2 * omega
        v_r = v + self.ell / 2 * omega
        u_out = np.array([v_l, v_r])
        return u_out

    def draw(self, x, y, theta):
        """
        Finds points that draw a differential drive vehicle.

        The centre of the wheel axle is (x, y), the vehicle has orientation
        theta, and the vehicle's track length is ell.

        Returns X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C, where L is for the
        left wheel, R for the right wheel, B for the body, and C for the caster.
        """
        # Left and right wheels
        x_l, y_l = graphics.draw_rectangle(
            x - 0.5 * self.ell * np.sin(theta),
            y + 0.5 * self.ell * np.cos(theta),
            0.5 * self.ell,
            0.25 * self.ell,
            theta,
        )
        x_r, y_r = graphics.draw_rectangle(
            x + 0.5 * self.ell * np.sin(theta),
            y - 0.5 * self.ell * np.cos(theta),
            0.5 * self.ell,
            0.25 * self.ell,
            theta,
        )
        # Body
        x_bd, y_bd = graphics.draw_circle(x, y, self.ell)
        # Caster
        x_c, y_c = graphics.draw_circle(
            x + 0.5 * self.ell * np.cos(theta),
            y + 0.5 * self.ell * np.sin(theta),
            0.125 * self.ell,
        )
        # Return the arrays of points
        return x_l, y_l, x_r, y_r, x_bd, y_bd, x_c, y_c

    def animate(self, x, dt, save_ani=False, filename="animate_diff_drive.gif"):
        """Create an animation of a differential drive vehicle.

        Returns animation object for array of vehicle positions x with time
        increments dt [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diff_drive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (left_wheel,) = ax.fill([], [], color="k")
        (right_wheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """Function that initializes the animation."""
            line.set_data([], [])
            left_wheel.set_xy(np.empty([5, 2]))
            right_wheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return line, left_wheel, right_wheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the differential drive vehicle
            x_l, y_l, x_r, y_r, x_b, y_b, x_c, y_c = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            left_wheel.set_xy(np.transpose([x_l, y_l]))
            right_wheel.set_xy(np.transpose([x_r, y_r]))
            body.set_xy(np.transpose([x_b, y_b]))
            castor.set_xy(np.transpose([x_c, y_c]))
            # Add the simulation time
            time_text.set_text(rf"$t$ = {k * dt:.1f} s")
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, left_wheel, right_wheel, body, castor, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / dt / 10))),
            init_func=init,
            interval=dt * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani:
            ani.save(filename, fps=min(1 / dt, 10))
        # Return the figure object
        return ani

    def animate_trajectory(
        self, x, xd, dt, save_ani=False, filename="animate_diff_drive.gif"
    ):
        """Create an animation of a differential drive vehicle with plots of
        actual and desired trajectories.

        Returns animation object for array of vehicle positions and desired
        positions x with time increments dt [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diff_drive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (desired,) = ax.plot([], [], "--C1")
        (line,) = ax.plot([], [], "C0")
        (left_wheel,) = ax.fill([], [], color="k")
        (right_wheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """Function that initializes the animation."""
            desired.set_data([], [])
            line.set_data([], [])
            left_wheel.set_xy(np.empty([5, 2]))
            right_wheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return desired, line, left_wheel, right_wheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the desired trajectory
            desired.set_data(xd[0, 0 : k + 1], xd[1, 0 : k + 1])
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the differential drive vehicle
            x_l, y_l, x_r, y_r, x_b, y_b, x_c, y_c = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            left_wheel.set_xy(np.transpose([x_l, y_l]))
            right_wheel.set_xy(np.transpose([x_r, y_r]))
            body.set_xy(np.transpose([x_b, y_b]))
            castor.set_xy(np.transpose([x_c, y_c]))
            # Add the simulation time
            time_text.set_text(rf"$t$ = {k * dt:.1f} s")
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return desired, line, left_wheel, right_wheel, body, castor, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / dt / 10))),
            init_func=init,
            interval=dt * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani:
            ani.save(filename, fps=min(1 / dt, 10))
        # Return the figure object
        return ani

    def animate_estimation(
        self,
        x,
        x_hat,
        p_cov_hat,
        alpha,
        dt,
        save_ani=False,
        filename="animate_diff_drive.gif",
    ):
        """Create an animation of a differential drive vehicle with plots of
        estimation uncertainty.

        Returns animation object for array of vehicle positions x with time
        increments dt [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diff_drive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (estimated,) = ax.plot([], [], "--C1")
        (line,) = ax.plot([], [], "C0")
        (left_wheel,) = ax.fill([], [], color="k")
        (right_wheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
        s2 = chi2.isf(alpha, 2)

        def init():
            """Function that initializes the animation."""
            estimated.set_data([], [])
            line.set_data([], [])
            left_wheel.set_xy(np.empty([5, 2]))
            right_wheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return estimated, line, left_wheel, right_wheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the desired trajectory
            estimated.set_data(x_hat[0, 0 : k + 1], x_hat[1, 0 : k + 1])
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the differential drive vehicle
            x_l, y_l, x_r, y_r, x_b, y_b, x_c, y_c = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            left_wheel.set_xy(np.transpose([x_l, y_l]))
            right_wheel.set_xy(np.transpose([x_r, y_r]))
            body.set_xy(np.transpose([x_b, y_b]))
            castor.set_xy(np.transpose([x_c, y_c]))
            # Compute eigenvalues and eigenvectors to find axes for covariance ellipse
            w_mat, v_mat = np.linalg.eig(p_cov_hat[0:2, 0:2, k])
            # Find the index of the largest and smallest eigenvalues
            j_max = np.argmax(w_mat)
            j_min = np.argmin(w_mat)
            ell = patches.Ellipse(
                (x_hat[0, k], x_hat[1, k]),
                2 * np.sqrt(s2 * w_mat[j_max]),
                2 * np.sqrt(s2 * w_mat[j_min]),
                angle=np.arctan2(v_mat[j_max, 1], v_mat[j_max, 0]) * 180 / np.pi,
                alpha=0.2,
                color="C1",
            )
            ax.add_artist(ell)
            # Add the simulation time
            time_text.set_text(rf"$t$ = {k * dt:.1f} s")
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return estimated, line, left_wheel, right_wheel, body, castor, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / dt / 10))),
            init_func=init,
            interval=dt * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani:
            ani.save(filename, fps=min(1 / dt, 10))
        # Return the figure object
        return ani


class Tricycle:
    """Tricycle or planar bicycle vehicle class.

    Parameters
    ----------
    ell_W : float
        The wheelbase of the vehicle [m].
    ell_T : float
        The vehicle's track length [m].
    """

    def __init__(self, ell_w, ell_t):
        """Constructor method."""
        self.ell_w = ell_w
        self.ell_t = ell_t

    def f(self, x, u):
        """Tricycle or planar bicycle vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = np.zeros(4)
        f[0] = u[0] * np.cos(x[2])
        f[1] = u[0] * np.sin(x[2])
        f[2] = u[0] * 1.0 / self.ell_w * np.tan(x[3])
        f[3] = u[1]
        return f

    def draw(self, x, y, theta, phi):
        """Finds points that draw a tricycle vehicle.

        The centre of the rear wheel axle is (x, y), the body has orientation
        theta, steering angle phi, wheelbase ell_W and track length ell_T.

        Returns X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B, where L is for the left
        wheel, R is for the right wheel, F is for the single front wheel, and
        BD is for the vehicle's body.
        """
        # Left and right back wheels
        x_l, y_l = graphics.draw_rectangle(
            x - 0.5 * self.ell_t * np.sin(theta),
            y + 0.5 * self.ell_t * np.cos(theta),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            theta,
        )
        x_r, y_r = graphics.draw_rectangle(
            x + 0.5 * self.ell_t * np.sin(theta),
            y - 0.5 * self.ell_t * np.cos(theta),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            theta,
        )
        # Front wheel
        x_f, y_f = graphics.draw_rectangle(
            x + self.ell_w * np.cos(theta),
            y + self.ell_w * np.sin(theta),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            theta + phi,
        )
        # Body
        x_bd, y_bd = graphics.draw_rectangle(
            x + self.ell_w / 2.0 * np.cos(theta),
            y + self.ell_w / 2.0 * np.sin(theta),
            2.0 * self.ell_w,
            2.0 * self.ell_t,
            theta,
        )
        # Return the arrays of points
        return x_l, y_l, x_r, y_r, x_f, y_f, x_bd, y_bd

    def animate(
        self,
        x,
        dt,
        save_ani=False,
        filename="animate_tricycle.gif",
    ):
        """Create an animation of a tricycle vehicle.

        Returns animation object for array of vehicle positions x with time
        increments dt [s], wheelbase ell_W [m], and track ell_T [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default 'animate_tricycle.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (left_wheel,) = ax.fill([], [], color="k")
        (right_wheel,) = ax.fill([], [], color="k")
        (front_wheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """A function that initializes the animation."""
            line.set_data([], [])
            left_wheel.set_xy(np.empty([5, 2]))
            right_wheel.set_xy(np.empty([5, 2]))
            front_wheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([5, 2]))
            time_text.set_text("")
            return line, left_wheel, right_wheel, front_wheel, body, time_text

        def movie(k):
            """The function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the tricycle vehicle
            x_l, y_l, x_r, y_r, x_f, y_f, x_b, y_b = self.draw(
                x[0, k], x[1, k], x[2, k], x[3, k]
            )
            left_wheel.set_xy(np.transpose([x_l, y_l]))
            right_wheel.set_xy(np.transpose([x_r, y_r]))
            front_wheel.set_xy(np.transpose([x_f, y_f]))
            body.set_xy(np.transpose([x_b, y_b]))
            # Add the simulation time
            time_text.set_text(rf"$t$ = {k * dt:.1f} s")
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell_w, x[0, k] + 10 * self.ell_w)
            ax.set_ylim(x[1, k] - 10 * self.ell_w, x[1, k] + 10 * self.ell_w)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, left_wheel, right_wheel, front_wheel, body, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / dt / 10))),
            init_func=init,
            interval=dt * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani:
            ani.save(filename, fps=min(1 / dt, 10))
        # Return the figure object
        return ani


class Ackermann:
    """Ackermann steered vehicle class.

    Parameters
    ----------
    ell_W : float
        The wheelbase of the vehicle [m].
    ell_T : float
        The vehicle's track length [m].
    """

    def __init__(self, ell_w, ell_t):
        """Constructor method."""
        self.ell_w = ell_w
        self.ell_t = ell_t

    def f(self, x, u):
        """Ackermann steered vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2
            The vehicle's speed and steering angle rate.

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = np.zeros(4)
        f[0] = u[0] * np.cos(x[2])
        f[1] = u[0] * np.sin(x[2])
        f[2] = u[0] * 1.0 / self.ell_w * np.tan(x[3])
        f[3] = u[1]
        return f

    def ackermann(self, x):
        """Computes the Ackermann steering angles.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).

        Returns
        -------
        ackermann_angles : ndarray of length 2
            The left and right wheel angles (phi_L, phi_R).
        """
        phi_l = np.arctan(
            2 * self.ell_w * np.tan(x[3]) / (2 * self.ell_w - self.ell_t * np.tan(x[3]))
        )
        phi_r = np.arctan(
            2 * self.ell_w * np.tan(x[3]) / (2 * self.ell_w + self.ell_t * np.tan(x[3]))
        )
        ackermann_angles = np.array([phi_l, phi_r])
        return ackermann_angles

    def draw(self, x, y, theta, phi_l, phi_r):
        """Finds points that draw an Ackermann steered (car-like) vehicle.

        The centre of the rear wheel axle is (x, y), the body has orientation
        theta, effective steering angle phi, wheelbase ell_w and track length
        ell_t.

        Returns x_bl, y_bl, x_br, y_br, x_fl, y_fl, x_fr, y_fr, x_bd, y_bd,
        where L denotes left, R denotes right, B denotes back, F denotes front,
        and BD denotes the vehicle's body.
        """
        # Left and right back wheels
        x_bl, y_bl = graphics.draw_rectangle(
            x - 0.5 * self.ell_t * np.sin(theta),
            y + 0.5 * self.ell_t * np.cos(theta),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            theta,
        )
        x_br, y_br = graphics.draw_rectangle(
            x + 0.5 * self.ell_t * np.sin(theta),
            y - 0.5 * self.ell_t * np.cos(theta),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            theta,
        )
        # Left and right front wheels
        x_fl, y_fl = graphics.draw_rectangle(
            x + self.ell_w * np.cos(theta) - self.ell_t / 2 * np.sin(theta),
            y + self.ell_w * np.sin(theta) + self.ell_t / 2 * np.cos(theta),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            theta + phi_l,
        )
        x_fr, y_fr = graphics.draw_rectangle(
            x + self.ell_w * np.cos(theta) + self.ell_t / 2 * np.sin(theta),
            y + self.ell_w * np.sin(theta) - self.ell_t / 2 * np.cos(theta),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            theta + phi_r,
        )
        # Body
        x_bd, y_bd = graphics.draw_rectangle(
            x + self.ell_w / 2.0 * np.cos(theta),
            y + self.ell_w / 2.0 * np.sin(theta),
            2.0 * self.ell_w,
            2.0 * self.ell_t,
            theta,
        )
        # Return the arrays of points
        return x_bl, y_bl, x_br, y_br, x_fl, y_fl, x_fr, y_fr, x_bd, y_bd

    def animate(
        self,
        x,
        dt,
        phi_l,
        phi_r,
        save_ani=False,
        filename="animate_ackermann.gif",
    ):
        """Create an animation of an Ackermann steered (car-like) vehicle.

        Returns animation object for array of vehicle positions x with time
        increments dt [s], wheelbase ell_w [m], and track ell_t [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default 'animate_ackermann.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (bl_wheel,) = ax.fill([], [], color="k")
        (br_wheel,) = ax.fill([], [], color="k")
        (fl_wheel,) = ax.fill([], [], color="k")
        (fr_wheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """A function that initializes the animation."""
            line.set_data([], [])
            bl_wheel.set_xy(np.empty([5, 2]))
            br_wheel.set_xy(np.empty([5, 2]))
            fl_wheel.set_xy(np.empty([5, 2]))
            fr_wheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([5, 2]))
            time_text.set_text("")
            return line, bl_wheel, br_wheel, fl_wheel, fr_wheel, body, time_text

        def movie(k):
            """The function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the Ackermann steered drive vehicle
            x_bl, y_bl, x_br, y_br, x_fl, y_fl, x_fr, y_fr, x_bd, y_bd = self.draw(
                x[0, k], x[1, k], x[2, k], phi_l[k], phi_r[k]
            )
            bl_wheel.set_xy(np.transpose([x_bl, y_bl]))
            br_wheel.set_xy(np.transpose([x_br, y_br]))
            fl_wheel.set_xy(np.transpose([x_fl, y_fl]))
            fr_wheel.set_xy(np.transpose([x_fr, y_fr]))
            body.set_xy(np.transpose([x_bd, y_bd]))
            # Add the simulation time
            time_text.set_text(f"$t$ = {k * dt:.1f} s")
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell_w, x[0, k] + 10 * self.ell_w)
            ax.set_ylim(x[1, k] - 10 * self.ell_w, x[1, k] + 10 * self.ell_w)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, bl_wheel, br_wheel, fl_wheel, fr_wheel, body, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / dt / 10))),
            init_func=init,
            interval=dt * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani:
            ani.save(filename, fps=min(1 / dt, 10))
        # Return the figure object
        return ani


class FourWheelSteered:
    """Four-wheel steered vehicle class.

    Parameters
    ----------
    ell_w : float
        The wheelbase of the vehicle [m].
    ell_t : float
        The vehicle's track length [m].
    """

    def __init__(self, ell_w, ell_t):
        """Constructor method."""
        self.ell_w = ell_w
        self.ell_t = ell_t

    def f(self, x, u):
        """Four-wheel steered vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2
            The vehicle's front wheel speed and steering angle rate.

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = np.zeros(4)
        f[0] = u[0] * np.cos(x[2]) * np.cos(x[3])
        f[1] = u[0] * np.sin(x[2]) * np.cos(x[3])
        f[2] = u[0] * 1.0 / (0.5 * self.ell_w) * np.sin(x[3])
        f[3] = u[1]
        return f

    def ackermann(self, x):
        """Computes the Ackermann steering angles.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).

        Returns
        -------
        ackermann_angles : ndarray of length 2
            The left and right wheel angles (phi_L, phi_R).
        """
        phi_l = np.arctan(
            2 * self.ell_w * np.tan(x[3]) / (2 * self.ell_w - self.ell_t * np.tan(x[3]))
        )
        phi_r = np.arctan(
            2 * self.ell_w * np.tan(x[3]) / (2 * self.ell_w + self.ell_t * np.tan(x[3]))
        )
        ackermann_angles = np.array([phi_l, phi_r])
        return ackermann_angles

    def draw(self, x):
        """Finds points that draw a four-wheel steered vehicle.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).

        The geometric centre of the vehicle is (x, y), the body has orientation
        theta, effective steering angle phi, wheelbase ell_W and track length
        ell_T.

        Returns x_bl, y_bl, x_br, y_br, x_fl, y_fl, x_fr, y_fr, x_bd, y_bd,
        where l denotes left, r denotes right, b denotes back, f denotes front,
        and bd denotes the vehicle's body.
        """
        # Left and right back wheels
        x_bl, y_bl = graphics.draw_rectangle(
            x[0] - 0.5 * self.ell_w * np.cos(x[2]) - 0.5 * self.ell_t * np.sin(x[2]),
            x[1] - 0.5 * self.ell_w * np.sin(x[2]) + 0.5 * self.ell_t * np.cos(x[2]),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            x[2] - self.ackermann(x)[0],
        )
        x_br, y_br = graphics.draw_rectangle(
            x[0] - 0.5 * self.ell_w * np.cos(x[2]) + 0.5 * self.ell_t * np.sin(x[2]),
            x[1] - 0.5 * self.ell_w * np.sin(x[2]) - 0.5 * self.ell_t * np.cos(x[2]),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            x[2] - self.ackermann(x)[1],
        )
        # Left and right front wheels
        x_fl, y_fl = graphics.draw_rectangle(
            x[0] + 0.5 * self.ell_w * np.cos(x[2]) - self.ell_t / 2 * np.sin(x[2]),
            x[1] + 0.5 * self.ell_w * np.sin(x[2]) + self.ell_t / 2 * np.cos(x[2]),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            x[2] + self.ackermann(x)[0],
        )
        x_fr, y_fr = graphics.draw_rectangle(
            x[0] + 0.5 * self.ell_w * np.cos(x[2]) + self.ell_t / 2 * np.sin(x[2]),
            x[1] + 0.5 * self.ell_w * np.sin(x[2]) - self.ell_t / 2 * np.cos(x[2]),
            0.5 * self.ell_t,
            0.25 * self.ell_t,
            x[2] + self.ackermann(x)[1],
        )
        # Body
        x_bd, y_bd = graphics.draw_rectangle(
            x[0],
            x[1],
            2.0 * self.ell_w,
            2.0 * self.ell_t,
            x[2],
        )
        # Return the arrays of points
        return x_bl, y_bl, x_br, y_br, x_fl, y_fl, x_fr, y_fr, x_bd, y_bd

    def animate(
        self,
        x,
        dt,
        save_ani=False,
        filename="animate_four_wheel_steered.gif",
    ):
        """Create an animation of an four-wheel steered vehicle.

        Returns animation object for array of vehicle positions x with time
        increments dt [s], wheelbase ell_w [m], and track ell_t [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default 'animate_four_wheel_steered.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (bl_wheel,) = ax.fill([], [], color="k")
        (br_wheel,) = ax.fill([], [], color="k")
        (fl_wheel,) = ax.fill([], [], color="k")
        (fr_wheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """A function that initializes the animation."""
            line.set_data([], [])
            bl_wheel.set_xy(np.empty([5, 2]))
            br_wheel.set_xy(np.empty([5, 2]))
            fl_wheel.set_xy(np.empty([5, 2]))
            fr_wheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([5, 2]))
            time_text.set_text("")
            return line, bl_wheel, br_wheel, fl_wheel, fr_wheel, body, time_text

        def movie(k):
            """The function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the Ackermann steered drive vehicle
            x_bl, y_bl, x_br, y_br, x_fl, y_fl, x_fr, y_fr, x_bd, y_bd = self.draw(
                x[:, k]
            )
            bl_wheel.set_xy(np.transpose([x_bl, y_bl]))
            br_wheel.set_xy(np.transpose([x_br, y_br]))
            fl_wheel.set_xy(np.transpose([x_fl, y_fl]))
            fr_wheel.set_xy(np.transpose([x_fr, y_fr]))
            body.set_xy(np.transpose([x_bd, y_bd]))
            # Add the simulation time
            time_text.set_text(f"$t$ = {k * dt:.1f} s")
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell_w, x[0, k] + 10 * self.ell_w)
            ax.set_ylim(x[1, k] - 10 * self.ell_w, x[1, k] + 10 * self.ell_w)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, bl_wheel, br_wheel, fl_wheel, fr_wheel, body, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / dt / 10))),
            init_func=init,
            interval=dt * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani:
            ani.save(filename, fps=min(1 / dt, 10))
        # Return the figure object
        return ani
