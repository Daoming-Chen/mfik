"""
Robot 模块：机器人运动学

该模块提供机器人URDF解析、正运动学和逆运动学功能。

子模块:
- urdf: URDF文件解析
- forward_kinematics: 基于PyTorch的正运动学计算
- inverse_kinematics: 数值迭代逆运动学求解器
"""

from .urdf import (
    parse_urdf,
    get_kinematic_chain,
    check_joint_limits,
    clip_to_limits,
    shortest_angular_distance,
    normalize_angles,
    interpolate_angles,
)

from .forward_kinematics import (
    ForwardKinematics,
    compute_fk,
)

from .inverse_kinematics import (
    InverseKinematics,
    solve_ik,
)

__all__ = [
    # URDF
    'parse_urdf',
    'get_kinematic_chain',
    'check_joint_limits',
    'clip_to_limits',
    'shortest_angular_distance',
    'normalize_angles',
    'interpolate_angles',
    # Forward Kinematics
    'ForwardKinematics',
    'compute_fk',
    # Inverse Kinematics
    'InverseKinematics',
    'solve_ik',
]

