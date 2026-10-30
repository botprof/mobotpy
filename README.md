# MoBotPy

MoBotPy is a small [Python](https://www.python.org) package together with a set of worked autonomous mobile robot simulation examples.  Each example can be run as a standalone script, in [IPython](https://ipython.org), or in [Jupyter](https://jupyter.org).

These examples are intended to support teaching and learning in state variable control and estimation for mobile robots with emphasis on mathematical clarity, reproducibility, and progressive engineering realism.

This repository provides:

1. A small reusable Python package called `mobotpy` containing core mobile robot models and basic numerical tools; and
2. A collection of worked examples illustrating fundamental robotics concepts.

## Getting Started

Use these commands to set up and run your first example.  First, create a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the MoBotPy package, which also installs the project's declared dependencies.

```bash
python3 -m pip install .
```

Then run a script, for example:

```bash
python3 examples/linear_systems/kinematic.py
```

### Using VS Code Interactive

To run examples interactively in VS Code:

1. Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions.
2. Open this repository and create the project environment by following the setup steps above.
3. In VS Code, select the `.venv` environment with **Python: Select Interpreter**.
4. Open an example file. Add `# %%` above a section of code to make it an interactive cell, then select **Run Cell**.

The project dependencies include `ipykernel`, which VS Code uses to run Python code in the Interactive window.

## Repository Structure

Important elements of the repository are described by the table below.

Folder | Description
------ | -----------
`/mobotpy` | MoBotPy package source code
`/examples/linear_systems` | Linear systems fundamentals
`/examples/robot_models` (coming soon) | Example mobile robot simulations
`/examples/robot_control` (coming soon) | Control system design examples
`/examples/state_estimation` (coming soon) | State estimation examples
`/docs` (coming soon) | Supplementary documentation

## MoBotPy Package

This small package is required to run all example code. It contains some reusable code that provides mobile robot models, plotting and animation scripts, and basic numerical integration.

Module Filename | Description
--------------- | -----------
[`integration.py`](mobotpy/integration.py) | Provides basic Runge-Kutta and Euler integration functions
[`models.py`](mobotpy/models.py) | Provides standard vehicle models, plotting, and animation methods
[`graphics.py`](mobotpy/graphics.py) | Provides some basic shape plotting functions

## Worked Examples

Worked simulation examples in Python are the reason for this repository. These are described (with links) in the sections that follow.

### Linear Systems & Control

These examples provide a review of some linear systems and control fundamentals.

Python Script Filename | Example Description
---------------------- | ----------------------
[`kinematic.py`](examples/linear_systems/kinematic.py) | Linear kinematic mobile robot
[`dynamic.py`](examples/linear_systems/dynamic.py) | Linear dynamic mobile robot
[`kinematic_control.py`](examples/linear_systems/kinematic_control.py) | Kinematic proportional control problem
[`dynamic_control.py`](examples/linear_systems/dynamic_control.py) | Control of a linear dynamic mobile robot
[`discrete_control.py`](examples/linear_systems/discrete_control.py) | Control of a linear dynamic mobile robot in discrete time
[`integral_control.py`](examples/linear_systems/integral_control.py) | Demonstrates integral control
[`dynamic_observer.py`](examples/linear_systems/dynamic_observer.py) | Luenberger observer example
[`combined_control.py`](examples/linear_systems/combined_control.py) | Controller combined with an observer
[`disturbance_observer.py`](examples/linear_systems/disturbance_observer.py) | Example of a disturbance observer
[`discrete_lqr.py`](examples/linear_systems/discrete_lqr.py) | Control of a linear mobile robot with LQR
[`linear_mpc.py`](examples/linear_systems/linear_mpc.py) | Basic MPC implementation
[`linear_mpc_cvxpy.py`](examples/linear_systems/linear_mpc_cvxpy.py) | Linear MPC with constraints using CVXPY

## How to Cite

If you use this repository to help you with a report, assignment, or publication, please cite:

> J. A. Marshall, “MoBotPy,” version 0.1.0, GitHub, 2026. [Online]. Available: <https://github.com/botprof/mobotpy>. [Accessed: Oct. 30, 2026].

BibTeX:

```bibtex
@misc{marshall2026mobotpy,
    author       = {Marshall, Joshua A.},
    title        = {{MoBotPy}},
    howpublished = {GitHub repository, version 0.1.0},
    year         = {2026},
    url          = {https://github.com/botprof/mobotpy},
    note         = {Accessed: Oct.~30, 2026}
}
```

## License

Source code examples in this repository are subject to an [MIT License](LICENSE).
