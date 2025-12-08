"""
数据加载器模块

提供 PyTorch DataLoader 配置和数据增强功能。
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable
from .dataset import IKFlowDataset


class IKFlowDataLoader:
    """
    IK Flow 数据加载器配置类
    
    封装 PyTorch DataLoader，提供标准化的配置选项。
    """
    
    @staticmethod
    def create(dataset: IKFlowDataset,
              batch_size: int = 32,
              shuffle: bool = True,
              num_workers: int = 0,
              pin_memory: bool = False,
              drop_last: bool = False,
              collate_fn: Optional[Callable] = None) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: IKFlowDataset 对象
            batch_size: 批量大小
            shuffle: 是否随机打乱
            num_workers: 数据加载进程数
            pin_memory: 是否使用 pin_memory (GPU 训练推荐)
            drop_last: 是否丢弃最后不完整的批次
            collate_fn: 自定义 collate 函数
            
        Returns:
            DataLoader 对象
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn
        )


def default_collate_fn(batch):
    """
    默认的 collate 函数
    
    将批量样本转换为张量字典。
    """
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        collated[key] = torch.stack([sample[key] for sample in batch])
    
    return collated


def augment_collate_fn(batch,
                       joint_noise_std: float = 0.01,
                       position_noise_std: float = 0.001):
    """
    带数据增强的 collate 函数
    
    在批量加载时添加随机噪声进行数据增强。
    
    Args:
        batch: 样本列表
        joint_noise_std: 关节角度噪声标准差
        position_noise_std: 位置噪声标准差
        
    Returns:
        增强后的批量数据
    """
    # 默认 collate
    collated = default_collate_fn(batch)
    
    # 添加噪声
    if joint_noise_std > 0:
        q_t_noise = torch.randn_like(collated['q_t']) * joint_noise_std
        collated['q_t'] = collated['q_t'] + q_t_noise
    
    if position_noise_std > 0:
        pos_noise = torch.randn_like(collated['target_pos']) * position_noise_std
        collated['target_pos'] = collated['target_pos'] + pos_noise
    
    return collated


def create_dataloader(dataset: IKFlowDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = False,
                     drop_last: bool = False,
                     augment: bool = False,
                     joint_noise_std: float = 0.01,
                     position_noise_std: float = 0.001) -> DataLoader:
    """
    便捷函数：创建数据加载器
    
    Args:
        dataset: IKFlowDataset 对象
        batch_size: 批量大小
        shuffle: 是否随机打乱
        num_workers: 数据加载进程数
        pin_memory: 是否使用 pin_memory
        drop_last: 是否丢弃最后不完整的批次
        augment: 是否使用数据增强
        joint_noise_std: 关节角度噪声标准差 (仅当 augment=True)
        position_noise_std: 位置噪声标准差 (仅当 augment=True)
        
    Returns:
        DataLoader 对象
    """
    if augment:
        # 使用增强 collate 函数
        collate_fn = lambda batch: augment_collate_fn(
            batch, joint_noise_std, position_noise_std
        )
    else:
        collate_fn = default_collate_fn
    
    return IKFlowDataLoader.create(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


class MultiEpochDataLoader:
    """
    多 epoch 数据加载器
    
    预生成多个 epoch 的迭代器，避免每个 epoch 重新初始化。
    适用于小数据集的快速训练。
    """
    
    def __init__(self, dataloader: DataLoader, num_epochs: int):
        """
        Args:
            dataloader: DataLoader 对象
            num_epochs: 预生成的 epoch 数量
        """
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epochs = []
        
        # 预生成 epochs
        for _ in range(num_epochs):
            self.epochs.append(list(dataloader))
    
    def __iter__(self):
        for epoch_data in self.epochs:
            for batch in epoch_data:
                yield batch
    
    def __len__(self):
        return len(self.epochs) * len(self.dataloader)
