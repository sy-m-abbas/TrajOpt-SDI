"""
Contact-Implicit Trajectory Optimization Package

A differentiable contact physics simulator with trajectory optimization
for planar manipulation tasks.
"""

from .qp_solver import ContactQPSolver
from .geometry import rot, closest_point_on_square_and_normal, contact_frame_and_J
from .dynamics import step_square, rollout
from .optimizer import TrajectoryOptimizer

__version__ = "0.1.0"
__all__ = [
    "ContactQPSolver",
    "rot",
    "closest_point_on_square_and_normal", 
    "contact_frame_and_J",
    "step_square",
    "rollout",
    "TrajectoryOptimizer"
]