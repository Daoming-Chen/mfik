"""
正运动学模块

基于PyTorch实现的可微分正运动学计算,支持批量计算和自动微分。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .urdf import KinematicChain


class ForwardKinematics:
    """
    正运动学计算器
    
    基于PyTorch实现的可微分FK,支持批量计算和GPU加速。
    """
    
    def __init__(self, chain: KinematicChain, device: str = 'cpu'):
        """
        初始化FK计算器
        
        Args:
            chain: 运动学链对象
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.chain = chain
        self.device = device
        self.n_joints = chain.n_joints
        
        # 预计算关节轴向和原点变换
        self._precompute_transforms()
    
    def _precompute_transforms(self):
        """预计算关节的固定变换"""
        # 关节轴向
        self.joint_axes = torch.zeros(self.n_joints, 3, device=self.device)
        # 关节原点(相对于父链接的变换)
        self.joint_origins = torch.zeros(self.n_joints, 4, 4, device=self.device)
        
        for i, joint in enumerate(self.chain.joints):
            # 关节轴
            self.joint_axes[i] = torch.tensor(joint.axis, dtype=torch.float32)
            
            # 原点变换矩阵
            if joint.origin is not None:
                self.joint_origins[i] = torch.tensor(joint.origin, dtype=torch.float32)
            else:
                self.joint_origins[i] = torch.eye(4)
    
    def compute(self, q: torch.Tensor, 
                link_name: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算正运动学
        
        Args:
            q: 关节角度, shape: [batch_size, n_joints] 或 [n_joints]
            link_name: 指定链接名称(None表示末端执行器)
            
        Returns:
            position: 位置 [batch_size, 3] 或 [3]
            quaternion: 四元数姿态 [batch_size, 4] 或 [4] (w, x, y, z)
        """
        # 确保q是2D
        if q.dim() == 1:
            q = q.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = q.shape[0]
        
        # 确保在正确的设备上
        q = q.to(self.device)
        
        # 计算到目标链接的变换矩阵
        if link_name is None:
            n_links = self.n_joints
        else:
            # 找到链接在链中的索引
            try:
                n_links = self.chain.link_names.index(link_name) + 1
            except ValueError:
                raise ValueError(f"Link {link_name} not found in kinematic chain")
        
        # 计算累积变换
        T = self._forward_kinematics(q, n_links)
        
        # 提取位置和姿态
        position = T[:, :3, 3]
        rotation = T[:, :3, :3]
        quaternion = self._rotation_matrix_to_quaternion(rotation)
        
        if squeeze_output:
            position = position.squeeze(0)
            quaternion = quaternion.squeeze(0)
        
        return position, quaternion
    
    def _forward_kinematics(self, q: torch.Tensor, n_links: int) -> torch.Tensor:
        """
        计算前n_links个关节的累积变换
        
        Args:
            q: 关节角度 [batch_size, n_joints]
            n_links: 要计算的链接数量
            
        Returns:
            T: 变换矩阵 [batch_size, 4, 4]
        """
        batch_size = q.shape[0]
        
        # 初始化为单位矩阵 (使用与q相同的dtype)
        T = torch.eye(4, device=self.device, dtype=q.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for i in range(min(n_links, self.n_joints)):
            # 关节原点变换
            T_origin = self.joint_origins[i].unsqueeze(0).expand(batch_size, -1, -1)
            # Ensure T_origin has correct dtype if modified externally, or rely on auto-casting
            if T_origin.dtype != q.dtype:
                T_origin = T_origin.to(dtype=q.dtype)
                
            T = torch.bmm(T, T_origin)
            
            # 关节旋转/平移
            joint_type = self.chain.joint_types[i]
            axis = self.joint_axes[i]
            # Ensure axis has correct dtype
            if axis.dtype != q.dtype:
                axis = axis.to(dtype=q.dtype)
                
            angle = q[:, i]
            
            if joint_type in ['revolute', 'continuous']:
                # 旋转关节
                T_joint = self._rotation_transform(axis, angle)
            elif joint_type == 'prismatic':
                # 平移关节
                T_joint = self._translation_transform(axis, angle)
            else:
                # 固定关节或其他类型,使用单位矩阵
                T_joint = torch.eye(4, device=self.device, dtype=q.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            
            T = torch.bmm(T, T_joint)
        
        return T
    
    def _rotation_transform(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        根据轴角表示计算旋转变换矩阵(Rodrigues公式)
        
        Args:
            axis: 旋转轴 [3]
            angle: 旋转角度 [batch_size]
            
        Returns:
            T: 旋转变换矩阵 [batch_size, 4, 4]
        """
        batch_size = angle.shape[0]
        
        # 归一化轴
        axis = axis / torch.norm(axis)
        
        # 反对称矩阵
        K = torch.zeros(3, 3, device=self.device, dtype=angle.dtype)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        
        # Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=self.device, dtype=angle.dtype)
        sin_angle = torch.sin(angle).view(-1, 1, 1)
        cos_angle = torch.cos(angle).view(-1, 1, 1)
        
        K2 = torch.mm(K, K)
        R = I + sin_angle * K + (1 - cos_angle) * K2
        
        # 构造4x4变换矩阵
        T = torch.eye(4, device=self.device, dtype=angle.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        T[:, :3, :3] = R
        
        return T
    
    def _translation_transform(self, axis: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """
        计算平移变换矩阵
        
        Args:
            axis: 平移轴 [3]
            distance: 平移距离 [batch_size]
            
        Returns:
            T: 平移变换矩阵 [batch_size, 4, 4]
        """
        batch_size = distance.shape[0]
        
        # 归一化轴
        axis = axis / torch.norm(axis)
        
        # 构造变换矩阵
        T = torch.eye(4, device=self.device, dtype=distance.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        translation = axis.unsqueeze(0) * distance.unsqueeze(1)
        T[:, :3, 3] = translation
        
        return T
    
    def _rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """
        将旋转矩阵转换为四元数
        
        Args:
            R: 旋转矩阵 [batch_size, 3, 3]
            
        Returns:
            q: 四元数 [batch_size, 4] (w, x, y, z)
        """
        batch_size = R.shape[0]
        q = torch.zeros(batch_size, 4, device=self.device, dtype=R.dtype)
        
        # Shepperd's method for numerical stability
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        
        # Case 1: trace > 0
        mask1 = trace > 0
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
        
        # Case 2: R[0,0] is maximum
        mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
        
        # Case 3: R[1,1] is maximum
        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
        
        # Case 4: R[2,2] is maximum
        mask4 = (~mask1) & (~mask2) & (~mask3)
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s
        
        # 归一化
        q = q / torch.norm(q, dim=1, keepdim=True)
        
        return q
    
    def compute_batch(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量计算正运动学(与compute相同,但强调批量处理)
        
        Args:
            q: 关节角度 [batch_size, n_joints]
            
        Returns:
            position: 位置 [batch_size, 3]
            quaternion: 四元数姿态 [batch_size, 4] (w, x, y, z)
        """
        return self.compute(q)


def compute_fk(chain: KinematicChain, q: Union[np.ndarray, torch.Tensor],
               device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    便捷函数:计算正运动学
    
    Args:
        chain: 运动学链对象
        q: 关节角度, shape: [batch_size, n_joints] 或 [n_joints]
        device: 计算设备
        
    Returns:
        position: 位置 [batch_size, 3] 或 [3]
        quaternion: 四元数姿态 [batch_size, 4] 或 [4]
    """
    fk = ForwardKinematics(chain, device)
    
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q).float()
    
    return fk.compute(q)

