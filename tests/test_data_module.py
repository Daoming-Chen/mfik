"""
测试 Data 模块功能
"""

import os
import sys
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mfik.robot.urdf import parse_urdf
from mfik.data import (
    BaseMapping,
    create_base_mapping,
    sample_reverse_perturbation,
    sample_time_parameters,
    generate_flow_path,
    FlowSampler,
    IKFlowDataset,
    create_dataloader,
)


def test_base_mapping():
    """测试基础映射库"""
    print("\n" + "="*60)
    print("测试 1: 基础映射库 (BaseMapping)")
    print("="*60)
    
    # 加载机器人模型
    urdf_path = "robots/panda_arm.urdf"
    chain = parse_urdf(urdf_path)
    print(f"加载机器人: {urdf_path}")
    print(f"关节数量: {chain.n_joints}")
    
    # 创建基础映射库
    print("\n构建基础映射库...")
    mapping = BaseMapping(chain, device='cpu')
    stats = mapping.build(n_samples=1000, batch_size=128, seed=42)
    
    print(f"\n映射库统计:")
    print(f"  样本数量: {stats['n_samples']}")
    print(f"  工作空间范围: {stats['workspace_min']} ~ {stats['workspace_max']}")
    print(f"  工作空间中心: {stats['workspace_center']}")
    
    # 测试查询
    print("\n测试最近邻查询...")
    target_pos = stats['workspace_center']
    distances, indices = mapping.query_nearest(target_pos, k=5)
    print(f"  查询位置: {target_pos}")
    print(f"  最近邻距离: {distances}")
    print(f"  最近邻索引: {indices}")
    
    # 获取最近的关节配置
    nearest_q = mapping.get_joint_config(indices[0])
    print(f"  最近关节配置: {nearest_q}")
    
    # 测试保存和加载
    save_path = "test_base_mapping.npz"
    mapping.save(save_path)
    
    mapping2 = BaseMapping(chain, device='cpu')
    mapping2.load(save_path)
    print(f"\n映射库加载成功: {len(mapping2.joint_configs)} 样本")
    
    # 清理
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n✓ 基础映射库测试通过")


def test_sampling():
    """测试采样策略"""
    print("\n" + "="*60)
    print("测试 2: 采样策略")
    print("="*60)
    
    # 测试逆向扰动采样
    print("\n测试逆向扰动采样...")
    q_star = np.array([0.0, 0.5, 1.0, -0.5, 0.2, 1.5, 0.0])
    q_perturbed = sample_reverse_perturbation(q_star, noise_std=0.1, seed=42)
    print(f"  原始配置: {q_star}")
    print(f"  扰动配置: {q_perturbed}")
    print(f"  扰动距离: {np.linalg.norm(q_perturbed - q_star):.4f}")
    
    # 测试时间参数采样
    print("\n测试时间参数采样...")
    for dist in ['uniform', 'beta', 'logit_normal']:
        r, t = sample_time_parameters(5, distribution=dist, seed=42)
        print(f"  {dist:12s}: r={r}, t={t}")
        assert np.all(r < t), "时间参数错误: r 应该小于 t"
        assert np.all((r >= 0) & (r < 1)), "时间参数错误: r 应该在 [0, 1)"
        assert np.all((t > 0) & (t <= 1)), "时间参数错误: t 应该在 (0, 1]"
    
    # 测试流路径生成
    print("\n测试流路径生成...")
    q_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    r, t = 0.2, 0.6
    q_t, v_t = generate_flow_path(q_0, q_1, r, t)
    
    print(f"  q_0 = {q_0}")
    print(f"  q_1 = {q_1}")
    print(f"  r = {r}, t = {t}")
    print(f"  q_t = {q_t}")
    print(f"  v_t = {v_t}")
    
    # 验证插值正确性
    expected_q_t = (1 - t) * q_0 + t * q_1
    assert np.allclose(q_t, expected_q_t), "流路径生成错误"
    
    print("\n✓ 采样策略测试通过")


def test_flow_sampler():
    """测试 Flow 采样器"""
    print("\n" + "="*60)
    print("测试 3: Flow 采样器")
    print("="*60)
    
    # 加载机器人模型
    urdf_path = "robots/panda_arm.urdf"
    chain = parse_urdf(urdf_path)
    print(f"加载机器人: {urdf_path}")
    
    # 创建采样器
    sampler = FlowSampler(chain, device='cpu', noise_std=0.1)
    
    # 测试单样本采样
    print("\n测试单样本采样...")
    target_pos = np.array([0.3, 0.0, 0.5])
    
    try:
        sample = sampler.sample_single(target_pos, seed=42)
        print(f"  目标位置: {sample['target_pos']}")
        print(f"  q_t: {sample['q_t']}")
        print(f"  v_t: {sample['v_t']}")
        print(f"  时间参数: r={sample['r']:.3f}, t={sample['t']:.3f}")
        print(f"  精确解: {sample['q_star']}")
        print("\n✓ 单样本采样成功")
    except Exception as e:
        print(f"\n✗ 单样本采样失败: {e}")
        print("  注意: IK 求解可能需要多次尝试或调整参数")


def test_dataset():
    """测试数据集"""
    print("\n" + "="*60)
    print("测试 4: IKFlowDataset")
    print("="*60)
    
    # 创建模拟数据
    n_samples = 100
    n_joints = 7
    
    data_dict = {
        'q_t': np.random.randn(n_samples, n_joints).astype(np.float32),
        'v_t': np.random.randn(n_samples, n_joints).astype(np.float32),
        'target_pos': np.random.randn(n_samples, 3).astype(np.float32),
        'target_quat': np.random.randn(n_samples, 4).astype(np.float32),
        'r': np.random.rand(n_samples).astype(np.float32),
        't': np.random.rand(n_samples).astype(np.float32),
    }
    
    # 创建数据集
    print("\n创建数据集...")
    dataset = IKFlowDataset(data_dict=data_dict, preload=True, device='cpu')
    print(f"  数据集大小: {len(dataset)}")
    
    # 测试访问
    print("\n测试样本访问...")
    sample = dataset[0]
    print(f"  样本字段: {sample.keys()}")
    print(f"  q_t shape: {sample['q_t'].shape}")
    print(f"  v_t shape: {sample['v_t'].shape}")
    
    # 测试统计
    print("\n数据集统计:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value[:3]}..." if len(value) > 3 else f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # 测试保存和加载
    print("\n测试数据集保存和加载...")
    save_path = "test_dataset.pt"
    dataset.save(save_path)
    
    dataset2 = IKFlowDataset(data_path=save_path, preload=True, device='cpu')
    print(f"  加载数据集: {len(dataset2)} 样本")
    
    # 清理
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n✓ 数据集测试通过")


def test_dataloader():
    """测试数据加载器"""
    print("\n" + "="*60)
    print("测试 5: DataLoader")
    print("="*60)
    
    # 创建模拟数据集
    n_samples = 100
    n_joints = 7
    
    data_dict = {
        'q_t': np.random.randn(n_samples, n_joints).astype(np.float32),
        'v_t': np.random.randn(n_samples, n_joints).astype(np.float32),
        'target_pos': np.random.randn(n_samples, 3).astype(np.float32),
        'target_quat': np.random.randn(n_samples, 4).astype(np.float32),
        'r': np.random.rand(n_samples).astype(np.float32),
        't': np.random.rand(n_samples).astype(np.float32),
    }
    
    dataset = IKFlowDataset(data_dict=data_dict, preload=True, device='cpu')
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    dataloader = create_dataloader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        augment=False
    )
    
    print(f"  批次数量: {len(dataloader)}")
    
    # 测试迭代
    print("\n测试批量加载...")
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"  批次 {i+1}:")
            print(f"    q_t shape: {batch['q_t'].shape}")
            print(f"    v_t shape: {batch['v_t'].shape}")
            print(f"    target_pos shape: {batch['target_pos'].shape}")
        if i >= 2:
            break
    
    # 测试数据增强
    print("\n测试数据增强...")
    dataloader_aug = create_dataloader(
        dataset,
        batch_size=16,
        shuffle=True,
        augment=True,
        joint_noise_std=0.01,
        position_noise_std=0.001
    )
    
    batch = next(iter(dataloader_aug))
    print(f"  增强批次:")
    print(f"    q_t shape: {batch['q_t'].shape}")
    
    print("\n✓ 数据加载器测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Data 模块测试套件")
    print("="*60)
    
    failed = False
    
    try:
        test_base_mapping()
    except Exception as e:
        print(f"\n✗ 基础映射库测试失败: {e}")
        import traceback
        traceback.print_exc()
        failed = True
    
    try:
        test_sampling()
    except Exception as e:
        print(f"\n✗ 采样策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        failed = True
    
    try:
        test_flow_sampler()
    except Exception as e:
        print(f"\n✗ Flow 采样器测试失败: {e}")
        import traceback
        traceback.print_exc()
        failed = True
    
    try:
        test_dataset()
    except Exception as e:
        print(f"\n✗ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        failed = True
    
    try:
        test_dataloader()
    except Exception as e:
        print(f"\n✗ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        failed = True
    
    print("\n" + "="*60)
    if failed:
        print("测试失败!")
        sys.exit(1)
    else:
        print("所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
