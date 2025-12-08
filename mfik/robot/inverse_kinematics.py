"""
逆运动学模块

基于PyTorch实现的数值迭代IK求解器,支持雅可比伪逆法和阻尼最小二乘法。
使用自动微分(Autograd)计算雅可比矩阵,支持批量并行求解。
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
from .urdf import KinematicChain
from .forward_kinematics import ForwardKinematics


class OptimizationFailureError(Exception):
    """优化失败异常"""
    pass


class InverseKinematics:
    """
    逆运动学求解器
    
    使用数值迭代方法求解IK,支持:
    - 雅可比伪逆法 (Jacobian pseudo-inverse)
    - 阻尼最小二乘法 (Damped Least Squares / Levenberg-Marquardt)
    """
    
    def __init__(self, chain: KinematicChain, device: str = 'cpu'):
        """
        初始化IK求解器
        
        Args:
            chain: 运动学链对象
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.chain = chain
        self.device = device
        self.fk = ForwardKinematics(chain, device)
        self.n_joints = chain.n_joints
    
    def solve(self, 
              target_pos: torch.Tensor,
              target_quat: Optional[torch.Tensor] = None,
              q_init: Optional[torch.Tensor] = None,
              method: str = 'jacobian',  # Default to jacobian (pinv) as it's more stable for high precision
              max_iter: int = 500,
              tolerance: float = 1e-6,
              damping: float = 0.01,
              step_size: float = 0.5) -> Optional[torch.Tensor]:
        """
        求解逆运动学
        
        Args:
            target_pos: 目标位置 [3] 或 [batch_size, 3]
            target_quat: 目标四元数 [4] 或 [batch_size, 4] (w,x,y,z), None表示仅位置
            q_init: 初始关节角度 [n_joints] 或 [batch_size, n_joints]
            method: 求解方法 ('jacobian' 或 'dls'/'lma')
            max_iter: 最大迭代次数
            tolerance: 收敛阈值 (位置误差 m)
            damping: 阻尼系数 (用于DLS/LMA方法)
            step_size: 步长因子
            
        Returns:
            q: 求解的关节角度 [n_joints] 或 [batch_size, n_joints]
        """
        # 使用 float64 进行高精度计算
        dtype = torch.float64
        
        # 处理输入维度
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = target_pos.shape[0]
        # 转换目标到 float64
        target_pos = target_pos.to(device=self.device, dtype=dtype)
        
        if target_quat is not None:
            if target_quat.dim() == 1:
                target_quat = target_quat.unsqueeze(0)
            target_quat = target_quat.to(device=self.device, dtype=dtype)
            use_orientation = True
        else:
            use_orientation = False
        
        # 关节限位
        lower = torch.from_numpy(self.chain.lower_limits).to(device=self.device, dtype=dtype)
        upper = torch.from_numpy(self.chain.upper_limits).to(device=self.device, dtype=dtype)
        
        # 初始化关节角度
        if q_init is None:
            # 使用关节限位中心作为初始值，避免初始化在无效区域
            safe_lower = torch.where(torch.isinf(lower), -torch.pi, lower)
            safe_upper = torch.where(torch.isinf(upper), torch.pi, upper)
            mid_q = (safe_lower + safe_upper) / 2.0
            
            # 添加随机扰动
            q = mid_q.unsqueeze(0).expand(batch_size, -1) + \
                torch.randn(batch_size, self.n_joints, device=self.device, dtype=dtype) * 0.1
        else:
            if q_init.dim() == 1:
                q = q_init.unsqueeze(0).expand(batch_size, -1).clone()
            else:
                q = q_init.clone()
            q = q.to(device=self.device, dtype=dtype)
            
            # 如果初始值为全0, 尝试移动到安全区域
            if torch.all(q == 0):
                 safe_lower = torch.where(torch.isinf(lower), -torch.pi, lower)
                 safe_upper = torch.where(torch.isinf(upper), torch.pi, upper)
                 mid_q = (safe_lower + safe_upper) / 2.0
                 q = mid_q.unsqueeze(0).expand(batch_size, -1) + \
                     torch.randn(batch_size, self.n_joints, device=self.device, dtype=dtype) * 0.1
        
        # 保存 FK 原始 dtype
        old_fk_axes_dtype = self.fk.joint_axes.dtype
        old_fk_origins_dtype = self.fk.joint_origins.dtype
        
        # 临时将 FK 内部缓冲区转换为 float64
        self.fk.joint_axes = self.fk.joint_axes.to(dtype=dtype)
        self.fk.joint_origins = self.fk.joint_origins.to(dtype=dtype)
        
        try:
            # 迭代求解
            for iter_count in range(max_iter):
                q.requires_grad_(True)
                
                # 计算当前位姿
                current_pos, current_quat = self.fk.compute(q)
                
                # 计算误差
                if use_orientation:
                    error = self._compute_pose_error(current_pos, current_quat,
                                                     target_pos, target_quat)
                else:
                    error = self._compute_position_error(current_pos, target_pos)
                
                # 检查收敛
                with torch.no_grad():
                    error_norm = torch.norm(error, dim=1)
                    pos_error_norm = torch.norm(error[:, :3], dim=1)
                    if torch.all(pos_error_norm < tolerance):
                        break
                
                # 计算雅可比矩阵
                J_err = self._compute_jacobian_autograd(q, error)
                
                # 计算更新量
                with torch.no_grad():
                    if method == 'jacobian':
                        delta_q = self._jacobian_step(J_err, error)
                    elif method in ['lma', 'dls']:
                        delta_q = self._dls_step(J_err, error, damping)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    q = q - step_size * delta_q
                    q = torch.clamp(q, lower, upper)
                    
                q = q.detach()
                
        finally:
            # 恢复 FK dtype
            self.fk.joint_axes = self.fk.joint_axes.to(dtype=old_fk_axes_dtype)
            self.fk.joint_origins = self.fk.joint_origins.to(dtype=old_fk_origins_dtype)
        
        # 最终结果处理
        # Convert back to float32 (default)
        q = q.to(dtype=torch.float32)
        
        if squeeze_output:
            q = q.squeeze(0)
        
        return q
    
    def _compute_position_error(self, current_pos: torch.Tensor,
                                target_pos: torch.Tensor) -> torch.Tensor:
        return current_pos - target_pos

    def _compute_pose_error(self, current_pos: torch.Tensor, current_quat: torch.Tensor,
                           target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        # 位置误差
        pos_error = current_pos - target_pos
        
        # 姿态误差
        quat_error = self._quaternion_multiply(current_quat, 
                                                self._quaternion_conjugate(target_quat))
        ori_error = 2.0 * quat_error[:, 1:]
        
        # 合并
        error = torch.cat([pos_error, ori_error], dim=1)
        return error

    def _compute_jacobian_autograd(self, q: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        使用Autograd计算雅可比矩阵
        Returns: [batch_size, error_dim, n_joints]
        """
        batch_size = q.shape[0]
        error_dim = error.shape[1]
        n_joints = q.shape[1]
        
        jacobian = torch.zeros(batch_size, error_dim, n_joints, device=self.device, dtype=q.dtype)
        
        # 为每个误差分量计算梯度
        for i in range(error_dim):
            grad_output = torch.zeros_like(error)
            grad_output[:, i] = 1.0
            
            grads = torch.autograd.grad(outputs=error, inputs=q, 
                                      grad_outputs=grad_output,
                                      retain_graph=(i < error_dim - 1),
                                      create_graph=False,
                                      allow_unused=True)[0]
            
            if grads is not None:
                jacobian[:, i, :] = grads
                
        return jacobian
    
    def _jacobian_step(self, J: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        计算雅可比伪逆更新量 (Vectorized)
        solve J * dq = -error => dq = -pinv(J) * error
        Here we return pinv(J) * error, caller does subtraction.
        """
        # J: [B, E, N], error: [B, E]
        
        # torch.linalg.pinv supports batch
        J_pinv = torch.linalg.pinv(J) # [B, N, E]
        
        error = error.unsqueeze(2) # [B, E, 1]
        delta_q = torch.bmm(J_pinv, error).squeeze(2) # [B, N]
        
        return delta_q
    
    def _dls_step(self, J: torch.Tensor, error: torch.Tensor, damping: float) -> torch.Tensor:
        """
        计算DLS/LMA更新量 (Vectorized)
        (J^T J + lambda^2 I) dq = -J^T error
        We return dq solving this.
        """
        batch_size = J.shape[0]
        n_joints = J.shape[2]
        
        Jt = J.transpose(1, 2) # [B, N, E]
        JtJ = torch.bmm(Jt, J) # [B, N, N]
        
        # Damping term
        damping_matrix = (damping ** 2) * torch.eye(n_joints, device=self.device, dtype=J.dtype).unsqueeze(0)
        A = JtJ + damping_matrix # [B, N, N]
        
        # RHS: J^T * error
        error = error.unsqueeze(2) # [B, E, 1]
        g = torch.bmm(Jt, error) # [B, N, 1]
        
        # Solve linear system A * dq = g
        delta_q = torch.linalg.solve(A, g).squeeze(2) # [B, N]
        
        return delta_q

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=1)
    
    def _quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        q_conj = q.clone()
        q_conj[:, 1:] = -q_conj[:, 1:]
        return q_conj
    
    def solve_batch(self, target_pos: torch.Tensor,
                   target_quat: Optional[torch.Tensor] = None,
                   q_init: Optional[torch.Tensor] = None,
                   **kwargs) -> torch.Tensor:
        return self.solve(target_pos, target_quat, q_init, **kwargs)


def solve_ik(chain: KinematicChain,
             target_pos: Union[np.ndarray, torch.Tensor],
             target_quat: Optional[Union[np.ndarray, torch.Tensor]] = None,
             q_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
             method: str = 'dls',
             device: str = 'cpu',
             **kwargs) -> torch.Tensor:
    ik = InverseKinematics(chain, device)
    
    if isinstance(target_pos, np.ndarray):
        target_pos = torch.from_numpy(target_pos).float()
    if target_quat is not None and isinstance(target_quat, np.ndarray):
        target_quat = torch.from_numpy(target_quat).float()
    if q_init is not None and isinstance(q_init, np.ndarray):
        q_init = torch.from_numpy(q_init).float()
    
    return ik.solve(target_pos, target_quat, q_init, method=method, **kwargs)
