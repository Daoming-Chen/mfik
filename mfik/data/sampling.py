"""
采样策略模块

实现 MeanFlow 训练数据的采样策略:
1. 逆向扰动采样 (Reverse Perturbation Sampling)
2. 时间参数采样 (logit-normal 分布)
3. 线性插值流路径生成
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict
from scipy.stats import beta
from ..robot.urdf import KinematicChain
from ..robot.forward_kinematics import ForwardKinematics
from ..robot.inverse_kinematics import InverseKinematics


def sample_random_configurations(chain: KinematicChain, 
                                 n_samples: int,
                                 seed: Optional[int] = None) -> np.ndarray:
    """
    采样随机关节配置
    
    在关节限位范围内均匀采样随机关节配置，用于生成测试数据。
    
    Args:
        chain: 运动学链
        n_samples: 采样数量
        seed: 随机种子
        
    Returns:
        关节配置数组 [n_samples, n_joints]
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_joints = chain.n_joints
    lower = chain.lower_limits
    upper = chain.upper_limits
    
    # 在关节限位范围内均匀采样
    q_samples = np.random.uniform(lower, upper, size=(n_samples, n_joints))
    
    return q_samples


def sample_reverse_perturbation(q_star: np.ndarray,
                                noise_std: float = 0.1,
                                joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                seed: Optional[int] = None) -> np.ndarray:
    """
    逆向扰动采样
    
    从精确解 q* 出发，添加高斯噪声生成输入 q_input = q* + ε
    确保训练数据始终位于单一解的局部邻域内，避免多解边界问题。
    
    Args:
        q_star: 精确解 [n_joints] 或 [N, n_joints]
        noise_std: 噪声标准差 (rad)
        joint_limits: 关节限位 (lower, upper)
        seed: 随机种子
        
    Returns:
        扰动后的关节配置
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 添加高斯噪声
    noise = np.random.normal(0, noise_std, q_star.shape)
    q_perturbed = q_star + noise
    
    # 限位裁剪
    if joint_limits is not None:
        lower, upper = joint_limits
        q_perturbed = np.clip(q_perturbed, lower, upper)
    
    return q_perturbed


def sample_time_parameters(n_samples: int,
                          distribution: str = 'uniform',
                          beta_a: float = 2.0,
                          beta_b: float = 5.0,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    采样时间参数 (r, t)
    
    MeanFlow 需要从 t=0 到 t=1 的流路径数据。
    支持多种分布策略，偏重于采样接近起点的时间步。
    
    Args:
        n_samples: 采样数量
        distribution: 分布类型 ('uniform', 'beta', 'logit_normal')
        beta_a, beta_b: Beta 分布参数 (当 distribution='beta')
        seed: 随机种子
        
    Returns:
        r: 起始时间 [N] in [0, 1)
        t: 结束时间 [N] in (r, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    if distribution == 'uniform':
        # 均匀采样
        r = np.random.uniform(0, 1, n_samples)
        t = np.random.uniform(r, 1)
        
    elif distribution == 'beta':
        # Beta 分布 (偏向起点)
        r = beta.rvs(beta_a, beta_b, size=n_samples)
        # t 在 [r, 1] 内均匀采样
        delta_t = np.random.uniform(0, 1 - r)
        t = r + delta_t
        
    elif distribution == 'logit_normal':
        # Logit-Normal 分布
        mu, sigma = -1.0, 1.5
        z = np.random.normal(mu, sigma, n_samples)
        r = 1 / (1 + np.exp(-z))
        r = np.clip(r, 0, 0.99)
        
        delta_t = np.random.uniform(0, 1 - r)
        t = r + delta_t
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # 确保 r < t
    t = np.maximum(t, r + 1e-6)
    t = np.minimum(t, 1.0)
    
    return r, t


def generate_flow_path(q_0: np.ndarray,
                      q_1: np.ndarray,
                      r: float,
                      t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成线性插值流路径
    
    给定起点 q_0, 终点 q_1, 和时间参数 (r, t)，
    计算 q_r, q_t 以及速度 v_t = (q_1 - q_r) / (1 - r)
    
    Args:
        q_0: 起点关节配置 [n_joints] 或 [N, n_joints]
        q_1: 终点关节配置 [n_joints] 或 [N, n_joints]
        r: 起始时间 (scalar 或 [N])
        t: 当前时间 (scalar 或 [N])
        
    Returns:
        q_t: 当前时刻的关节配置
        v_t: 当前时刻的速度场
    """
    # 线性插值
    q_r = (1 - r) * q_0 + r * q_1
    q_t = (1 - t) * q_0 + t * q_1
    
    # 速度场: v_t = (q_1 - q_0)
    v_t = q_1 - q_0
    
    return q_t, v_t


class FlowSampler:
    """
    MeanFlow 数据采样器
    
    整合逆向扰动采样、时间参数采样和流路径生成，
    生成完整的训练样本。
    """
    
    def __init__(self,
                 chain: KinematicChain,
                 device: str = 'cpu',
                 noise_std: float = 0.1,
                 time_distribution: str = 'beta',
                 sampling_strategy: str = 'local'):
        """
        Args:
            chain: 运动学链
            device: 计算设备
            noise_std: 扰动噪声标准差
            time_distribution: 时间参数分布
            sampling_strategy: 采样策略 ('local' 或 'global')
        """
        self.chain = chain
        self.device = device
        self.noise_std = noise_std
        self.time_distribution = time_distribution
        self.sampling_strategy = sampling_strategy
        
        self.fk = ForwardKinematics(chain, device)
        self.ik = InverseKinematics(chain, device)
        
        self.joint_limits = (chain.lower_limits, chain.upper_limits)
    
    def sample_single(self,
                     target_pos: np.ndarray,
                     target_quat: Optional[np.ndarray] = None,
                     q_init: Optional[np.ndarray] = None,
                     seed: Optional[int] = None) -> Dict:
        """
        采样单个训练样本
        
        Args:
            target_pos: 目标位置 [3]
            target_quat: 目标姿态 [4] (可选)
            q_init: IK 初值 [n_joints] (可选)
            seed: 随机种子
            
        Returns:
            样本字典，包含:
                - q_t: 输入关节配置
                - v_t: 目标速度场
                - target_pos: 目标位置
                - target_quat: 目标姿态
                - r, t: 时间参数
                - q_star: 精确解 (用于验证)
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 1. 求解 IK 得到精确解 q*
        target_pos_t = torch.from_numpy(target_pos).float()
        target_quat_t = torch.from_numpy(target_quat).float() if target_quat is not None else None
        q_init_t = torch.from_numpy(q_init).float() if q_init is not None else None
        
        q_star = self.ik.solve(target_pos_t, target_quat_t, q_init_t, 
                              method='dls', tolerance=1e-6, max_iter=500)
        
        if q_star is None:
            raise RuntimeError("IK 求解失败")
        
        q_star = q_star.detach().cpu().numpy()
        
        # 2. 逆向扰动采样得到 q_0
        if self.sampling_strategy == 'global':
            # 全局采样: 50% 概率使用均匀随机采样，50% 概率使用大噪声扰动
            if np.random.random() < 0.5:
                q_0 = sample_random_configurations(self.chain, 1, seed=None)[0]
            else:
                q_0 = sample_reverse_perturbation(q_star, self.noise_std, 
                                                 self.joint_limits, seed)
        else:
            # 局部采样
            q_0 = sample_reverse_perturbation(q_star, self.noise_std, 
                                             self.joint_limits, seed)
        
        # 3. 采样时间参数 (r, t)
        r, t = sample_time_parameters(1, distribution=self.time_distribution, seed=seed)
        r, t = r[0], t[0]
        
        # 4. 生成流路径
        q_t, v_t = generate_flow_path(q_0, q_star, r, t)
        
        # 5. 计算 q_t 对应的位姿
        q_t_tensor = torch.from_numpy(q_t).float().to(self.device)
        with torch.no_grad():
            pos_t, quat_t = self.fk.compute(q_t_tensor)
        pos_t = pos_t.cpu().numpy()
        quat_t = quat_t.cpu().numpy()
        
        return {
            'q_t': q_t.astype(np.float32),
            'v_t': v_t.astype(np.float32),
            'target_pos': target_pos.astype(np.float32),
            'target_quat': (target_quat if target_quat is not None else quat_t).astype(np.float32),
            'r': np.float32(r),
            't': np.float32(t),
            'q_star': q_star.astype(np.float32),
            'pos_t': pos_t.astype(np.float32),
            'quat_t': quat_t.astype(np.float32),
        }
    
    def sample_batch(self,
                    target_positions: np.ndarray,
                    target_quaternions: Optional[np.ndarray] = None,
                    q_inits: Optional[np.ndarray] = None,
                    batch_size: int = 32,
                    verbose: bool = True) -> Dict:
        """
        批量采样训练样本
        
        Args:
            target_positions: 目标位置 [N, 3]
            target_quaternions: 目标姿态 [N, 4] (可选)
            q_inits: IK 初值 [N, n_joints] (可选)
            batch_size: IK 求解批量大小
            verbose: 是否打印进度
            
        Returns:
            样本字典 (所有字段都是 [N, ...] 数组)
        """
        n_samples = len(target_positions)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # 初始化结果列表
        results = {
            'q_t': [],
            'v_t': [],
            'target_pos': [],
            'target_quat': [],
            'r': [],
            't': [],
            'q_star': [],
            'pos_t': [],
            'quat_t': [],
        }
        
        if verbose:
            print(f"批量采样: {n_samples} 样本, {n_batches} 批次...")
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            batch_len = end - start
            
            # 1. Generate q_0 first (for global strategy)
            if self.sampling_strategy == 'global':
                # 50% random q_0, 50% perturbed q_star (but we don't have q_star yet)
                # Wait, if we want q_star to be close to q_0, we must generate q_0 first.
                # But for "perturbed" strategy, we need q_star first.
                # So we split the batch.
                
                mask = np.random.random(batch_len) < 0.5
                q_0_batch = np.zeros((batch_len, self.chain.n_joints))
                q_star_batch = np.zeros((batch_len, self.chain.n_joints))
                
                # Case A: Random q_0 -> Solve IK -> q_star (Closest solution)
                if np.any(mask):
                    n_random = np.sum(mask)
                    q_0_random = sample_random_configurations(self.chain, n_random)
                    q_0_batch[mask] = q_0_random
                    
                    # Solve IK using q_0 as init
                    pos_random = torch.from_numpy(target_positions[start:end][mask]).float()
                    quat_random = torch.from_numpy(target_quaternions[start:end][mask]).float() if target_quaternions is not None else None
                    q_init_random = torch.from_numpy(q_0_random).float()
                    
                    q_star_random = self.ik.solve_batch(pos_random, quat_random, q_init_random,
                                                      method='dls', tolerance=1e-6, max_iter=500)
                    q_star_batch[mask] = q_star_random.detach().cpu().numpy()

                # Case B: Solve IK (random init) -> q_star -> Perturb -> q_0 (Local exploration)
                if np.any(~mask):
                    n_local = np.sum(~mask)
                    pos_local = torch.from_numpy(target_positions[start:end][~mask]).float()
                    quat_local = torch.from_numpy(target_quaternions[start:end][~mask]).float() if target_quaternions is not None else None
                    q_init_local = torch.from_numpy(q_inits[start:end][~mask]).float() if q_inits is not None else None
                    
                    q_star_local = self.ik.solve_batch(pos_local, quat_local, q_init_local,
                                                     method='dls', tolerance=1e-6, max_iter=500)
                    q_star_local_np = q_star_local.detach().cpu().numpy()
                    q_star_batch[~mask] = q_star_local_np
                    
                    q_0_local = sample_reverse_perturbation(q_star_local_np, self.noise_std, self.joint_limits)
                    q_0_batch[~mask] = q_0_local

            else:
                # Local strategy: Solve IK -> q_star -> Perturb -> q_0
                pos_batch = torch.from_numpy(target_positions[start:end]).float()
                quat_batch = torch.from_numpy(target_quaternions[start:end]).float() if target_quaternions is not None else None
                q_init_batch = torch.from_numpy(q_inits[start:end]).float() if q_inits is not None else None
                
                q_star_batch = self.ik.solve_batch(pos_batch, quat_batch, q_init_batch,
                                                  method='dls', tolerance=1e-6, max_iter=500)
                q_star_batch = q_star_batch.detach().cpu().numpy()
                
                q_0_batch = sample_reverse_perturbation(q_star_batch, self.noise_std, 
                                                       self.joint_limits)
            
            # 批量时间参数
            
            # 批量时间参数
            r_batch, t_batch = sample_time_parameters(batch_len, 
                                                     distribution=self.time_distribution)
            
            # 批量流路径
            r_expanded = r_batch[:, np.newaxis]
            t_expanded = t_batch[:, np.newaxis]
            q_t_batch, v_t_batch = generate_flow_path(q_0_batch, q_star_batch, 
                                                     r_expanded, t_expanded)
            
            # 批量 FK
            q_t_tensor = torch.from_numpy(q_t_batch).float().to(self.device)
            with torch.no_grad():
                pos_t_batch, quat_t_batch = self.fk.compute(q_t_tensor)
            pos_t_batch = pos_t_batch.cpu().numpy()
            quat_t_batch = quat_t_batch.cpu().numpy()
            
            # 存储结果
            results['q_t'].append(q_t_batch)
            results['v_t'].append(v_t_batch)
            results['target_pos'].append(target_positions[start:end])
            results['target_quat'].append(
                target_quaternions[start:end] if target_quaternions is not None else quat_t_batch
            )
            results['r'].append(r_batch)
            results['t'].append(t_batch)
            results['q_star'].append(q_star_batch)
            results['pos_t'].append(pos_t_batch)
            results['quat_t'].append(quat_t_batch)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  处理: {i+1}/{n_batches} 批次")
        
        # 合并结果
        for key in results:
            results[key] = np.concatenate(results[key], axis=0).astype(np.float32)
        
        if verbose:
            print(f"采样完成: {len(results['q_t'])} 样本")
        
        return results
