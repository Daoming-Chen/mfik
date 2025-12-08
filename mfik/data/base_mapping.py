"""
基础映射库模块

在关节空间进行均匀采样，计算对应的末端位姿，构建 KD-Tree 索引。
用于快速查找给定目标位姿的最近邻关节配置，作为数值优化的初值。
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict
from scipy.spatial import cKDTree
from ..robot.urdf import KinematicChain
from ..robot.forward_kinematics import ForwardKinematics


class BaseMapping:
    """
    基础映射库: 关节空间 -> 末端位姿
    
    通过均匀采样关节空间，计算对应的末端位姿，并构建 KD-Tree 索引
    用于快速查找给定目标位姿的最近邻关节配置。
    """
    
    def __init__(self, chain: KinematicChain, device: str = 'cpu'):
        """
        Args:
            chain: 运动学链
            device: 计算设备
        """
        self.chain = chain
        self.device = device
        self.fk = ForwardKinematics(chain, device)
        
        # 数据存储
        self.joint_configs = None  # [N, n_joints]
        self.positions = None      # [N, 3]
        self.quaternions = None    # [N, 4]
        self.kdtree = None        # KD-Tree for position
        
    def build(self, 
              n_samples: int = 100000,
              batch_size: int = 1024,
              seed: Optional[int] = None) -> Dict:
        """
        构建基础映射库
        
        Args:
            n_samples: 采样数量
            batch_size: 批量计算大小
            seed: 随机种子
            
        Returns:
            统计信息字典
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        print(f"构建基础映射库: {n_samples} 样本...")
        
        # 获取关节限位
        lower = self.chain.lower_limits
        upper = self.chain.upper_limits
        n_joints = self.chain.n_joints
        
        # 关节空间均匀采样
        joint_configs = []
        for i in range(n_joints):
            if np.isinf(lower[i]) or np.isinf(upper[i]):
                # 无限制关节，使用 [-π, π]
                samples = np.random.uniform(-np.pi, np.pi, n_samples)
            else:
                # 有限制关节，在限位内均匀采样
                samples = np.random.uniform(lower[i], upper[i], n_samples)
            joint_configs.append(samples)
        
        joint_configs = np.stack(joint_configs, axis=1)  # [N, n_joints]
        
        # 批量计算 FK
        positions = []
        quaternions = []
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        print(f"批量计算 FK: {n_batches} 批次...")
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            
            q_batch = torch.from_numpy(joint_configs[start:end]).float().to(self.device)
            
            with torch.no_grad():
                pos, quat = self.fk.compute(q_batch)
            
            positions.append(pos.cpu().numpy())
            quaternions.append(quat.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                print(f"  处理: {i+1}/{n_batches} 批次")
        
        # 合并结果
        self.joint_configs = joint_configs
        self.positions = np.concatenate(positions, axis=0)
        self.quaternions = np.concatenate(quaternions, axis=0)
        
        # 构建 KD-Tree (仅使用位置)
        print("构建 KD-Tree 索引...")
        self.kdtree = cKDTree(self.positions)
        
        # 统计工作空间范围
        stats = {
            'n_samples': n_samples,
            'n_joints': n_joints,
            'workspace_min': self.positions.min(axis=0),
            'workspace_max': self.positions.max(axis=0),
            'workspace_center': self.positions.mean(axis=0),
            'workspace_std': self.positions.std(axis=0),
        }
        
        print(f"基础映射库构建完成!")
        print(f"  工作空间范围: {stats['workspace_min']} ~ {stats['workspace_max']}")
        print(f"  工作空间中心: {stats['workspace_center']}")
        
        return stats
    
    def query_nearest(self, 
                      target_pos: np.ndarray,
                      k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        查询最近邻关节配置
        
        Args:
            target_pos: 目标位置 [3] 或 [N, 3]
            k: 返回最近的 k 个邻居
            
        Returns:
            distances: 距离 [k] 或 [N, k]
            indices: 索引 [k] 或 [N, k]
        """
        if self.kdtree is None:
            raise RuntimeError("基础映射库未构建，请先调用 build()")
        
        if target_pos.ndim == 1:
            target_pos = target_pos.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        distances, indices = self.kdtree.query(target_pos, k=k)
        
        if squeeze_output:
            distances = distances.squeeze(0)
            indices = indices.squeeze(0)
        
        return distances, indices
    
    def get_joint_config(self, indices: np.ndarray) -> np.ndarray:
        """
        获取指定索引的关节配置
        
        Args:
            indices: 索引数组
            
        Returns:
            关节配置 [n_joints] 或 [N, n_joints]
        """
        if self.joint_configs is None:
            raise RuntimeError("基础映射库未构建")
        
        return self.joint_configs[indices]
    
    def get_pose(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定索引的位姿
        
        Args:
            indices: 索引数组
            
        Returns:
            position: 位置 [3] 或 [N, 3]
            quaternion: 四元数 [4] 或 [N, 4]
        """
        if self.positions is None or self.quaternions is None:
            raise RuntimeError("基础映射库未构建")
        
        return self.positions[indices], self.quaternions[indices]
    
    def save(self, filepath: str):
        """保存基础映射库到文件"""
        np.savez_compressed(
            filepath,
            joint_configs=self.joint_configs,
            positions=self.positions,
            quaternions=self.quaternions
        )
        print(f"基础映射库已保存到: {filepath}")
    
    def load(self, filepath: str):
        """从文件加载基础映射库"""
        data = np.load(filepath)
        self.joint_configs = data['joint_configs']
        self.positions = data['positions']
        self.quaternions = data['quaternions']
        
        # 重建 KD-Tree
        print("重建 KD-Tree 索引...")
        self.kdtree = cKDTree(self.positions)
        print(f"基础映射库已加载: {len(self.joint_configs)} 样本")


def create_base_mapping(chain: KinematicChain,
                       n_samples: int = 100000,
                       device: str = 'cpu',
                       seed: Optional[int] = None) -> BaseMapping:
    """
    便捷函数：创建并构建基础映射库
    
    Args:
        chain: 运动学链
        n_samples: 采样数量
        device: 计算设备
        seed: 随机种子
        
    Returns:
        构建好的基础映射库
    """
    mapping = BaseMapping(chain, device)
    mapping.build(n_samples=n_samples, seed=seed)
    return mapping
