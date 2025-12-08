"""
URDF 解析模块

提供URDF文件解析和运动学链提取功能。
依赖于现有的urdf.py库,提供简洁的接口。
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import sys

# 导入现有的URDF解析库
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from urdf import URDF as _URDF


class KinematicChain:
    """运动学链数据结构
    
    存储从基座到末端执行器的关节序列和相关信息。
    
    Attributes:
        joints: 关节对象列表
        joint_names: 关节名称列表
        joint_types: 关节类型列表 ('revolute', 'prismatic', 'continuous', etc.)
        joint_axes: 关节轴向列表
        joint_limits: 关节限位 (lower, upper) 元组列表
        link_names: 链接名称列表
        base_link: 基座链接名称
        end_link: 末端链接名称
    """
    
    def __init__(self, urdf: _URDF, base_link: Optional[str] = None, 
                 end_link: Optional[str] = None):
        """
        从URDF对象构建运动学链
        
        Args:
            urdf: URDF对象
            base_link: 基座链接名称(None表示使用URDF的base_link)
            end_link: 末端链接名称(None表示使用第一个end_link)
        """
        self.urdf = urdf
        self.base_link = base_link or urdf.base_link.name
        self.end_link = end_link or (urdf.end_links[0].name if urdf.end_links else None)
        
        # 只包含actuated joints(排除fixed和mimic joints)
        self.joints = urdf.actuated_joints
        self.joint_names = [j.name for j in self.joints]
        self.joint_types = [j.joint_type for j in self.joints]
        self.joint_axes = [j.axis for j in self.joints]
        
        # 提取关节限位
        self.joint_limits = []
        for joint in self.joints:
            if joint.limit is not None:
                lower = joint.limit.lower if joint.limit.lower is not None else -np.inf
                upper = joint.limit.upper if joint.limit.upper is not None else np.inf
                self.joint_limits.append((lower, upper))
            else:
                # continuous joint 没有限位
                self.joint_limits.append((-np.inf, np.inf))
        
        # 链接名称
        self.link_names = [j.child for j in self.joints]
        if self.joints:
            self.link_names.insert(0, self.joints[0].parent)
    
    @property
    def n_joints(self) -> int:
        """关节数量"""
        return len(self.joints)
    
    @property
    def lower_limits(self) -> np.ndarray:
        """关节下限数组"""
        return np.array([l for l, u in self.joint_limits])
    
    @property
    def upper_limits(self) -> np.ndarray:
        """关节上限数组"""
        return np.array([u for l, u in self.joint_limits])
    
    def print_tree(self):
        """打印运动学链树形结构"""
        print(f"Kinematic Chain: {self.base_link} -> {self.end_link}")
        print(f"Number of joints: {self.n_joints}\n")
        
        for i, joint in enumerate(self.joints):
            print(f"Joint {i}: {joint.name}")
            print(f"  Type: {joint.joint_type}")
            print(f"  Parent link: {joint.parent}")
            print(f"  Child link: {joint.child}")
            print(f"  Axis: {joint.axis}")
            if joint.limit:
                print(f"  Limits: [{joint.limit.lower}, {joint.limit.upper}]")
            else:
                print(f"  Limits: continuous")
            print()


def parse_urdf(urdf_path: str) -> KinematicChain:
    """
    解析URDF文件
    
    Args:
        urdf_path: URDF文件路径
        
    Returns:
        KinematicChain: 运动学链对象
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: URDF格式错误
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    try:
        urdf = _URDF.load(urdf_path)
        return KinematicChain(urdf)
    except Exception as e:
        raise ValueError(f"Failed to parse URDF file: {e}")


def get_kinematic_chain(urdf_path: str, base_link: Optional[str] = None,
                       end_link: Optional[str] = None) -> KinematicChain:
    """
    获取运动学链
    
    Args:
        urdf_path: URDF文件路径
        base_link: 基座链接名称
        end_link: 末端链接名称
        
    Returns:
        KinematicChain: 运动学链对象
    """
    urdf = _URDF.load(urdf_path)
    return KinematicChain(urdf, base_link, end_link)


def check_joint_limits(q: Union[np.ndarray, torch.Tensor], 
                      chain: KinematicChain) -> Union[np.ndarray, torch.Tensor]:
    """
    检查关节角度是否在限位内
    
    Args:
        q: 关节角度, shape: [..., n_joints]
        chain: 运动学链对象
        
    Returns:
        布尔数组, shape: [..., n_joints], True表示在限位内
    """
    lower = chain.lower_limits
    upper = chain.upper_limits
    
    if isinstance(q, torch.Tensor):
        lower = torch.from_numpy(lower).to(q.device, q.dtype)
        upper = torch.from_numpy(upper).to(q.device, q.dtype)
        return (q >= lower) & (q <= upper)
    else:
        q = np.asarray(q)
        return (q >= lower) & (q <= upper)


def clip_to_limits(q: Union[np.ndarray, torch.Tensor], 
                  chain: KinematicChain) -> Union[np.ndarray, torch.Tensor]:
    """
    将关节角度裁剪到限位范围内
    
    Args:
        q: 关节角度, shape: [..., n_joints]
        chain: 运动学链对象
        
    Returns:
        裁剪后的关节角度, shape: [..., n_joints]
    """
    lower = chain.lower_limits
    upper = chain.upper_limits
    
    if isinstance(q, torch.Tensor):
        lower = torch.from_numpy(lower).to(q.device, q.dtype)
        upper = torch.from_numpy(upper).to(q.device, q.dtype)
        return torch.clamp(q, lower, upper)
    else:
        q = np.asarray(q)
        return np.clip(q, lower, upper)


def shortest_angular_distance(q1: Union[float, np.ndarray, torch.Tensor],
                              q2: Union[float, np.ndarray, torch.Tensor]
                              ) -> Union[float, np.ndarray, torch.Tensor]:
    """
    计算两个角度之间的最短角度差(考虑周期性)
    
    适用于旋转关节,结果在 [-π, π] 范围内。
    
    Args:
        q1: 起始角度
        q2: 目标角度
        
    Returns:
        最短角度差 (q2 - q1), 在 [-π, π] 范围内
    """
    if isinstance(q1, torch.Tensor) or isinstance(q2, torch.Tensor):
        if not isinstance(q1, torch.Tensor):
            q1 = torch.tensor(q1, dtype=torch.float32)
        if not isinstance(q2, torch.Tensor):
            q2 = torch.tensor(q2, dtype=torch.float32)
        
        diff = q2 - q1
        # 归一化到 [-π, π]
        return torch.atan2(torch.sin(diff), torch.cos(diff))
    else:
        if isinstance(q1, (int, float)):
            q1 = np.array(q1)
        if isinstance(q2, (int, float)):
            q2 = np.array(q2)
        
        diff = q2 - q1
        # 归一化到 [-π, π]
        return np.arctan2(np.sin(diff), np.cos(diff))


def normalize_angles(q: Union[np.ndarray, torch.Tensor]
                    ) -> Union[np.ndarray, torch.Tensor]:
    """
    将角度归一化到 [-π, π] 范围
    
    Args:
        q: 角度, shape: [...]
        
    Returns:
        归一化后的角度, shape: [...]
    """
    if isinstance(q, torch.Tensor):
        return torch.atan2(torch.sin(q), torch.cos(q))
    else:
        q = np.asarray(q)
        return np.arctan2(np.sin(q), np.cos(q))


def interpolate_angles(q0: Union[np.ndarray, torch.Tensor],
                      q1: Union[np.ndarray, torch.Tensor],
                      t: Union[float, np.ndarray, torch.Tensor]
                      ) -> Union[np.ndarray, torch.Tensor]:
    """
    在两个关节配置之间插值(沿最短弧)
    
    对于旋转关节,使用最短弧插值;
    对于移动关节,使用线性插值。
    
    Args:
        q0: 起始关节角度, shape: [..., n_joints]
        q1: 目标关节角度, shape: [..., n_joints]
        t: 插值参数 in [0, 1], 标量或shape: [...]
        
    Returns:
        插值后的关节角度, shape: [..., n_joints]
    """
    # 计算最短角度差
    delta_q = shortest_angular_distance(q0, q1)
    
    # 线性插值
    if isinstance(q0, torch.Tensor):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=q0.dtype, device=q0.device)
        # 确保t的shape可以广播
        if t.dim() == 0:
            return q0 + t * delta_q
        else:
            # t: [batch], delta_q: [batch, n_joints]
            return q0 + t.unsqueeze(-1) * delta_q
    else:
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        t = np.asarray(t)
        
        if t.ndim == 0:
            return q0 + t * delta_q
        else:
            return q0 + t[..., np.newaxis] * delta_q

