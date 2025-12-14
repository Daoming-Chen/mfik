#!/usr/bin/env python3
"""
数据生成脚本

生成 MeanFlow IK 训练数据集，支持多机器人模型和可配置参数。

用法:
    python scripts/generate_data.py --robot panda --n-samples 10000 --output data/panda_10k.pt
    python scripts/generate_data.py --robot ur10 --n-samples 100000 --output data/ur10_100k.pt --device cuda
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mfik.robot.urdf import parse_urdf
from mfik.data import (
    BaseMapping,
    FlowSampler,
    IKFlowDataset,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成 MeanFlow IK 训练数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 机器人模型
    parser.add_argument(
        '--robot',
        type=str,
        default='panda',
        choices=['panda', 'ur10'],
        help='机器人模型名称'
    )

    parser.add_argument(
        '--urdf-path',
        type=str,
        default=None,
        help='URDF 文件路径 (如果不指定，使用默认路径)'
    )

    # 数据生成参数
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='生成的样本数量'
    )

    parser.add_argument(
        '--base-mapping-samples',
        type=int,
        default=100000,
        help='基础映射库的样本数量'
    )

    parser.add_argument(
        '--noise-std',
        type=float,
        default=0.1,
        help='逆向扰动采样的噪声标准差 (rad)'
    )

    parser.add_argument(
        '--time-distribution',
        type=str,
        default='beta',
        choices=['uniform', 'beta', 'logit_normal'],
        help='时间参数分布'
    )

    # 批量处理参数
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='IK 求解批量大小'
    )

    # 输出参数
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径 (.pt 或 .npz)'
    )

    parser.add_argument(
        '--save-base-mapping',
        type=str,
        default=None,
        help='保存基础映射库到指定路径 (可选)'
    )

    parser.add_argument(
        '--load-base-mapping',
        type=str,
        default=None,
        help='从指定路径加载基础映射库 (可选，加速数据生成)'
    )

    # 设备和随机种子
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='计算设备'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    # 其他选项
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    return parser.parse_args()


def get_urdf_path(robot_name: str, custom_path: str = None) -> str:
    """
    获取 URDF 文件路径

    Args:
        robot_name: 机器人名称
        custom_path: 自定义路径

    Returns:
        URDF 文件路径
    """
    if custom_path is not None:
        return custom_path

    # 默认路径
    default_paths = {
        'panda': 'robots/panda_arm.urdf',
        'ur10': 'robots/ur10.urdf',
    }

    return default_paths[robot_name]


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("="*70)
    print("MeanFlow IK 数据生成脚本")
    print("="*70)
    print(f"\n配置:")
    print(f"  机器人: {args.robot}")
    print(f"  样本数量: {args.n_samples}")
    print(f"  噪声标准差: {args.noise_std}")
    print(f"  时间分布: {args.time_distribution}")
    print(f"  设备: {args.device}")
    print(f"  输出: {args.output}")

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n警告: CUDA 不可用，切换到 CPU")
        args.device = 'cpu'

    # 加载机器人模型
    print("\n" + "-"*70)
    print("步骤 1: 加载机器人模型")
    print("-"*70)
    urdf_path = get_urdf_path(args.robot, args.urdf_path)
    print(f"URDF 路径: {urdf_path}")

    if not os.path.exists(urdf_path):
        print(f"错误: URDF 文件不存在: {urdf_path}")
        sys.exit(1)

    chain = parse_urdf(urdf_path)
    print(f"加载成功: {chain.n_joints} 个关节")
    print(f"关节名称: {[j.name for j in chain.joints]}")

    # 创建或加载基础映射库
    print("\n" + "-"*70)
    print("步骤 2: 构建/加载基础映射库")
    print("-"*70)

    mapping = BaseMapping(chain, device=args.device)

    if args.load_base_mapping is not None and os.path.exists(args.load_base_mapping):
        print(f"从文件加载: {args.load_base_mapping}")
        mapping.load(args.load_base_mapping)
    else:
        print(f"构建新的基础映射库: {args.base_mapping_samples} 样本")
        mapping.build(
            n_samples=args.base_mapping_samples,
            batch_size=args.batch_size * 4,
            seed=args.seed
        )

        # 保存基础映射库 (如果指定)
        if args.save_base_mapping is not None:
            os.makedirs(os.path.dirname(args.save_base_mapping), exist_ok=True)
            mapping.save(args.save_base_mapping)

    # 创建采样器
    print("\n" + "-"*70)
    print("步骤 3: 创建 Flow 采样器")
    print("-"*70)
    sampler = FlowSampler(
        chain,
        device=args.device,
        noise_std=args.noise_std,
        time_distribution=args.time_distribution
    )
    print(f"采样器配置:")
    print(f"  噪声标准差: {args.noise_std}")
    print(f"  时间分布: {args.time_distribution}")

    # 生成目标位姿
    print("\n" + "-"*70)
    print("步骤 4: 生成目标位姿")
    print("-"*70)

    # 从基础映射库中随机选择目标位姿
    indices = np.random.choice(len(mapping.positions), size=args.n_samples, replace=True)
    target_positions = mapping.positions[indices]
    target_quaternions = mapping.quaternions[indices]
    q_inits = mapping.joint_configs[indices]

    print(f"生成 {args.n_samples} 个目标位姿")
    print(f"工作空间范围:")
    print(f"  X: [{target_positions[:, 0].min():.3f}, {target_positions[:, 0].max():.3f}]")
    print(f"  Y: [{target_positions[:, 1].min():.3f}, {target_positions[:, 1].max():.3f}]")
    print(f"  Z: [{target_positions[:, 2].min():.3f}, {target_positions[:, 2].max():.3f}]")

    # 批量采样生成训练数据
    print("\n" + "-"*70)
    print("步骤 5: 批量采样生成训练数据")
    print("-"*70)

    try:
        samples = sampler.sample_batch(
            target_positions,
            target_quaternions,
            q_inits,
            batch_size=args.batch_size,
            verbose=args.verbose
        )

        print(f"\n采样完成!")
        print(f"  样本数量: {len(samples['q_t'])}")
        print(f"  数据字段: {list(samples.keys())}")
    except Exception as e:
        print(f"\n错误: 采样失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 创建数据集并保存
    print("\n" + "-"*70)
    print("步骤 6: 创建数据集并保存")
    print("-"*70)

    dataset = IKFlowDataset(data_dict=samples, preload=False, device='cpu')

    print(f"数据集统计:")
    stats = dataset.get_statistics()
    print(f"  样本数量: {stats['n_samples']}")
    print(f"  关节维度: {stats['n_joints']}")
    print(f"  q_t 均值: {stats['q_t_mean'][:3]}...")
    print(f"  q_t 标准差: {stats['q_t_std'][:3]}...")
    print(f"  目标位置范围: {stats['target_pos_min']} ~ {stats['target_pos_max']}")
    print(f"  时间参数 r 均值: {stats['r_mean']:.4f}")
    print(f"  时间参数 t 均值: {stats['t_mean']:.4f}")

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存数据集
    dataset.save(args.output, compress=True)

    # 计算文件大小
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n文件大小: {file_size_mb:.2f} MB")

    print("\n" + "="*70)
    print("数据生成完成!")
    print("="*70)
    print(f"\n输出文件: {args.output}")
    print(f"\n使用方法:")
    print(f"  from mfik.data import IKFlowDataset")
    print(f"  dataset = IKFlowDataset(data_path='{args.output}')")
    print(f"  print(len(dataset))  # {args.n_samples}")


if __name__ == "__main__":
    main()
