"""
Python module graphics.py for drawing basic shapes.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
URL: https://github.com/botprof/mobotpy
"""

import numpy as np


def draw_circle(x, y, radius, increment=10):
    """Finds points that approximate a circle as a regular polygon.

    The regular polygon has centre (x, y), a radius, and the function computes
    a point every angular increment [deg] (default 10).
    """
    angles = np.deg2rad(np.arange(0.0, 360.0, increment))
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)
    x_coords += x
    y_coords += y
    return x_coords, y_coords


def draw_rectangle(x, y, length, width, angle):
    """Finds points that draw a rectangle.

    The rectangle has centre (x, y), a length, width, and angle [rad].
    """
    l = 0.5 * length
    w = 0.5 * width
    vertices = np.array([[-l, -l, l, l, -l], [-w, w, w, -w, -w]])
    rotation_matrix = np.array(
        [[np.cos(angle), np.sin(-angle)], [np.sin(angle), np.cos(angle)]]
    )
    vertices = rotation_matrix @ vertices
    x_coords = vertices[0, :] + x
    y_coords = vertices[1, :] + y
    return x_coords, y_coords
