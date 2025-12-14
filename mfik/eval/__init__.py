"""
Eval Module - General-purpose inference and evaluation tools.

This module provides version-agnostic inference interfaces, evaluation metrics,
and visualization tools for IK solvers.
"""

from .inference import IKSolver
from .metrics import compute_pose_error, compute_success_rate, measure_latency

__all__ = [
    "IKSolver",
    "compute_pose_error",
    "compute_success_rate",
    "measure_latency",
]
