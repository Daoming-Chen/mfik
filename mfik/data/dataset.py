"""
数据集模块

提供 PyTorch Dataset 类，用于加载和缓存 MeanFlow IK 训练数据。
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Dict, List


class IKFlowDataset(Dataset):
    """
    IK Flow 数据集
    
    支持从 .pt 或 .npz 文件加载数据，提供批量访问和缓存机制。
    """
    
    def __init__(self,
                 data_path: Optional[str] = None,
                 data_dict: Optional[Dict] = None,
                 preload: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            data_path: 数据文件路径 (.pt 或 .npz)
            data_dict: 直接提供的数据字典 (用于内存数据)
            preload: 是否预加载到内存
            device: 数据存储设备 (用于预加载)
        """
        super().__init__()
        
        self.data_path = data_path
        self.preload = preload
        self.device = device
        
        # 数据字段
        self.q_t = None        # 输入关节配置 [N, n_joints]
        self.v_t = None        # 目标速度场 [N, n_joints]
        self.target_pos = None # 目标位置 [N, 3]
        self.target_quat = None # 目标姿态 [N, 4]
        self.r = None          # 起始时间 [N]
        self.t = None          # 当前时间 [N]
        
        # 可选字段
        self.q_star = None     # 精确解 (用于验证)
        self.pos_t = None      # 当前位置
        self.quat_t = None     # 当前姿态
        
        # 加载数据
        if data_dict is not None:
            self._load_from_dict(data_dict)
        elif data_path is not None:
            self._load_from_file(data_path)
        else:
            raise ValueError("必须提供 data_path 或 data_dict")
        
        self.n_samples = len(self.q_t)
    
    def _load_from_dict(self, data_dict: Dict):
        """从字典加载数据"""
        # 必需字段
        self.q_t = self._to_tensor(data_dict['q_t'])
        self.v_t = self._to_tensor(data_dict['v_t'])
        self.target_pos = self._to_tensor(data_dict['target_pos'])
        self.target_quat = self._to_tensor(data_dict['target_quat'])
        self.r = self._to_tensor(data_dict['r'])
        self.t = self._to_tensor(data_dict['t'])
        
        # 可选字段
        if 'q_star' in data_dict:
            self.q_star = self._to_tensor(data_dict['q_star'])
        if 'pos_t' in data_dict:
            self.pos_t = self._to_tensor(data_dict['pos_t'])
        if 'quat_t' in data_dict:
            self.quat_t = self._to_tensor(data_dict['quat_t'])
    
    def _load_from_file(self, filepath: str):
        """从文件加载数据"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        ext = os.path.splitext(filepath)[1]
        
        if ext == '.pt':
            # PyTorch 格式 (推荐)
            data = torch.load(filepath, map_location='cpu', weights_only=False)
            self._load_from_dict(data)
        
        elif ext == '.npz':
            # NumPy 压缩格式
            data = np.load(filepath)
            data_dict = {key: data[key] for key in data.keys()}
            self._load_from_dict(data_dict)
        
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        print(f"数据集已加载: {filepath}")
        print(f"  样本数量: {len(self.q_t)}")
        print(f"  关节维度: {self.q_t.shape[1]}")
    
    def _to_tensor(self, data):
        """将数据转换为 Tensor"""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            tensor = data.float()
        else:
            tensor = torch.tensor(data).float()
        
        if self.preload:
            tensor = tensor.to(self.device)
        
        return tensor
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            样本字典，包含:
                - q_t: 输入关节配置
                - v_t: 目标速度场
                - target_pos: 目标位置
                - target_quat: 目标姿态
                - r: 起始时间
                - t: 当前时间
        """
        sample = {
            'q_t': self.q_t[idx],
            'v_t': self.v_t[idx],
            'target_pos': self.target_pos[idx],
            'target_quat': self.target_quat[idx],
            'r': self.r[idx],
            't': self.t[idx],
        }
        
        # 可选字段
        if self.q_star is not None:
            sample['q_star'] = self.q_star[idx]
        if self.pos_t is not None:
            sample['pos_t'] = self.pos_t[idx]
        if self.quat_t is not None:
            sample['quat_t'] = self.quat_t[idx]
        
        return sample
    
    def save(self, filepath: str, compress: bool = True):
        """
        保存数据集到文件
        
        Args:
            filepath: 保存路径 (.pt 或 .npz)
            compress: 是否压缩 (仅对 .npz 有效)
        """
        ext = os.path.splitext(filepath)[1]
        
        # 准备数据字典
        data_dict = {
            'q_t': self.q_t.cpu().numpy(),
            'v_t': self.v_t.cpu().numpy(),
            'target_pos': self.target_pos.cpu().numpy(),
            'target_quat': self.target_quat.cpu().numpy(),
            'r': self.r.cpu().numpy(),
            't': self.t.cpu().numpy(),
        }
        
        if self.q_star is not None:
            data_dict['q_star'] = self.q_star.cpu().numpy()
        if self.pos_t is not None:
            data_dict['pos_t'] = self.pos_t.cpu().numpy()
        if self.quat_t is not None:
            data_dict['quat_t'] = self.quat_t.cpu().numpy()
        
        if ext == '.pt':
            # PyTorch 格式
            torch.save(data_dict, filepath)
        
        elif ext == '.npz':
            # NumPy 格式
            if compress:
                np.savez_compressed(filepath, **data_dict)
            else:
                np.savez(filepath, **data_dict)
        
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        print(f"数据集已保存: {filepath}")
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'n_samples': self.n_samples,
            'n_joints': self.q_t.shape[1],
            'q_t_mean': self.q_t.mean(dim=0).cpu().numpy(),
            'q_t_std': self.q_t.std(dim=0).cpu().numpy(),
            'v_t_mean': self.v_t.mean(dim=0).cpu().numpy(),
            'v_t_std': self.v_t.std(dim=0).cpu().numpy(),
            'target_pos_min': self.target_pos.min(dim=0)[0].cpu().numpy(),
            'target_pos_max': self.target_pos.max(dim=0)[0].cpu().numpy(),
            'r_mean': self.r.mean().item(),
            't_mean': self.t.mean().item(),
        }
        
        return stats


def create_dataset_from_samples(samples: Dict,
                                preload: bool = True,
                                device: str = 'cpu') -> IKFlowDataset:
    """
    从采样结果创建数据集
    
    Args:
        samples: FlowSampler.sample_batch() 的返回值
        preload: 是否预加载到内存
        device: 数据存储设备
        
    Returns:
        IKFlowDataset 对象
    """
    return IKFlowDataset(data_dict=samples, preload=preload, device=device)


def merge_datasets(datasets: List[IKFlowDataset]) -> IKFlowDataset:
    """
    合并多个数据集
    
    Args:
        datasets: 数据集列表
        
    Returns:
        合并后的数据集
    """
    if len(datasets) == 0:
        raise ValueError("数据集列表为空")
    
    if len(datasets) == 1:
        return datasets[0]
    
    # 合并所有字段
    merged_data = {}
    
    # 必需字段
    for key in ['q_t', 'v_t', 'target_pos', 'target_quat', 'r', 't']:
        merged_data[key] = torch.cat([ds.__dict__[key] for ds in datasets], dim=0)
    
    # 可选字段
    if datasets[0].q_star is not None:
        merged_data['q_star'] = torch.cat([ds.q_star for ds in datasets], dim=0)
    if datasets[0].pos_t is not None:
        merged_data['pos_t'] = torch.cat([ds.pos_t for ds in datasets], dim=0)
    if datasets[0].quat_t is not None:
        merged_data['quat_t'] = torch.cat([ds.quat_t for ds in datasets], dim=0)
    
    return IKFlowDataset(data_dict=merged_data, preload=datasets[0].preload, 
                        device=datasets[0].device)
