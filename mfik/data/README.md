# Data 模块

Data 模块提供 MeanFlow IK 训练数据的生成、索引和加载功能。

## 模块结构

```
mfik/data/
├── __init__.py          # 模块导出
├── base_mapping.py      # 基础映射库（关节空间 -> 位姿空间）
├── sampling.py          # 采样策略（逆向扰动、时间参数、流路径）
├── dataset.py           # PyTorch Dataset 类
├── loader.py            # DataLoader 配置
└── README.md            # 本文档
```

## 核心组件

### 1. BaseMapping - 基础映射库

**功能**: 在关节空间均匀采样，计算对应的末端位姿，构建 KD-Tree 索引。

**用途**: 快速查找给定目标位姿的最近邻关节配置，作为 IK 求解的初值。

**示例**:
```python
from mfik.robot.urdf import parse_urdf
from mfik.data import BaseMapping

# 加载机器人模型
chain = parse_urdf('robots/panda_arm.urdf')

# 创建并构建基础映射库
mapping = BaseMapping(chain, device='cpu')
mapping.build(n_samples=100000, batch_size=1024, seed=42)

# 查询最近邻
target_pos = np.array([0.3, 0.0, 0.5])
distances, indices = mapping.query_nearest(target_pos, k=5)
nearest_q = mapping.get_joint_config(indices[0])

# 保存和加载
mapping.save('base_mapping.npz')
mapping2 = BaseMapping(chain)
mapping2.load('base_mapping.npz')
```

### 2. FlowSampler - 流采样器

**功能**: 整合逆向扰动采样、时间参数采样和流路径生成，生成完整的 MeanFlow 训练样本。

**采样策略**:
- **逆向扰动采样**: 从精确解 q* 添加高斯噪声生成输入，确保数据在单一解的局部邻域内
- **时间参数采样**: 支持 uniform、beta、logit-normal 分布
- **流路径生成**: 线性插值生成从 q_0 到 q_1 的流路径

**示例**:
```python
from mfik.data import FlowSampler

# 创建采样器
sampler = FlowSampler(
    chain,
    device='cpu',
    noise_std=0.1,
    time_distribution='beta'
)

# 单样本采样
target_pos = np.array([0.3, 0.0, 0.5])
sample = sampler.sample_single(target_pos, seed=42)

# 批量采样
samples = sampler.sample_batch(
    target_positions,    # [N, 3]
    target_quaternions,  # [N, 4]
    q_inits,            # [N, n_joints]
    batch_size=32,
    verbose=True
)
```

**样本格式**:
```python
{
    'q_t': [n_joints],        # 输入关节配置
    'v_t': [n_joints],        # 目标速度场
    'target_pos': [3],        # 目标位置
    'target_quat': [4],       # 目标姿态（四元数）
    'r': scalar,              # 起始时间
    't': scalar,              # 当前时间
    'q_star': [n_joints],     # 精确解（用于验证）
    'pos_t': [3],            # 当前位置
    'quat_t': [4],           # 当前姿态
}
```

### 3. IKFlowDataset - 数据集类

**功能**: PyTorch Dataset 类，支持从 .pt 或 .npz 文件加载数据。

**特性**:
- 支持预加载到内存或 GPU
- 提供数据统计信息
- 支持保存和加载
- 可合并多个数据集

**示例**:
```python
from mfik.data import IKFlowDataset

# 从文件加载
dataset = IKFlowDataset(
    data_path='data/panda_10k.pt',
    preload=True,
    device='cpu'
)

# 从字典创建
dataset = IKFlowDataset(
    data_dict=samples,
    preload=True,
    device='cpu'
)

# 访问样本
sample = dataset[0]
print(len(dataset))

# 获取统计信息
stats = dataset.get_statistics()

# 保存数据集
dataset.save('output.pt', compress=True)
```

### 4. DataLoader - 数据加载器

**功能**: 封装 PyTorch DataLoader，提供标准化配置和数据增强。

**示例**:
```python
from mfik.data import create_dataloader

# 创建基础 DataLoader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    augment=False
)

# 创建带数据增强的 DataLoader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    augment=True,
    joint_noise_std=0.01,
    position_noise_std=0.001
)

# 训练循环
for batch in dataloader:
    q_t = batch['q_t']         # [batch_size, n_joints]
    v_t = batch['v_t']         # [batch_size, n_joints]
    target_pos = batch['target_pos']  # [batch_size, 3]
    # ...
```

## 数据生成流程

### 使用脚本生成数据

项目提供了 `scripts/generate_data.py` 脚本，用于生成训练数据集。

**基本用法**:
```bash
# 生成 Panda 机器人的 10k 样本数据集
python scripts/generate_data.py \
    --robot panda \
    --n-samples 10000 \
    --output data/panda_10k.pt

# 生成 UR10 机器人的 100k 样本数据集（使用 GPU）
python scripts/generate_data.py \
    --robot ur10 \
    --n-samples 100000 \
    --output data/ur10_100k.pt \
    --device cuda
```

**高级选项**:
```bash
python scripts/generate_data.py \
    --robot panda \
    --n-samples 100000 \
    --base-mapping-samples 200000 \
    --noise-std 0.15 \
    --time-distribution logit_normal \
    --batch-size 64 \
    --output data/panda_100k.pt \
    --save-base-mapping data/panda_base_mapping.npz \
    --device cuda \
    --seed 42 \
    --verbose
```

**参数说明**:
- `--robot`: 机器人模型名称 (panda, ur10)
- `--n-samples`: 生成的样本数量
- `--base-mapping-samples`: 基础映射库的样本数量（默认 100000）
- `--noise-std`: 逆向扰动噪声标准差（默认 0.1 rad）
- `--time-distribution`: 时间参数分布 (uniform, beta, logit_normal)
- `--batch-size`: IK 求解批量大小（默认 32）
- `--save-base-mapping`: 保存基础映射库到指定路径
- `--load-base-mapping`: 从指定路径加载基础映射库（加速数据生成）
- `--device`: 计算设备 (cpu, cuda)
- `--seed`: 随机种子
- `--verbose`: 显示详细输出

### 手动生成数据

如果需要更灵活的控制，可以手动编写数据生成代码：

```python
from mfik.robot.urdf import parse_urdf
from mfik.data import BaseMapping, FlowSampler, IKFlowDataset

# 1. 加载机器人模型
chain = parse_urdf('robots/panda_arm.urdf')

# 2. 构建基础映射库
mapping = BaseMapping(chain, device='cpu')
mapping.build(n_samples=100000, batch_size=1024)

# 3. 生成目标位姿
n_samples = 10000
indices = np.random.choice(len(mapping.positions), size=n_samples)
target_positions = mapping.positions[indices]
target_quaternions = mapping.quaternions[indices]
q_inits = mapping.joint_configs[indices]

# 4. 创建采样器并生成数据
sampler = FlowSampler(chain, device='cpu', noise_std=0.1)
samples = sampler.sample_batch(
    target_positions,
    target_quaternions,
    q_inits,
    batch_size=32
)

# 5. 创建数据集并保存
dataset = IKFlowDataset(data_dict=samples)
dataset.save('data/my_dataset.pt')
```

## 数据格式

### .pt 格式 (推荐)

PyTorch 原生格式，加载速度快，支持 GPU 直接加载。

```python
# 保存
torch.save(data_dict, 'dataset.pt')

# 加载
data = torch.load('dataset.pt')
```

### .npz 格式

NumPy 压缩格式，兼容性好，文件大小较小。

```python
# 保存
np.savez_compressed('dataset.npz', **data_dict)

# 加载
data = np.load('dataset.npz')
```

## 性能建议

### 基础映射库

- **样本数量**: 建议 100k-200k，覆盖工作空间
- **批量大小**: GPU 可用时使用 1024-4096
- **保存复用**: 构建一次，多次使用

### 数据生成

- **批量大小**: CPU 32-64，GPU 128-256
- **IK 容差**: 默认 1e-6 足够精确
- **噪声标准差**: 0.1 rad 适合大多数机器人

### 数据加载

- **预加载**: 小数据集（< 1GB）推荐预加载到内存
- **Pin Memory**: GPU 训练时启用
- **Workers**: CPU 训练时使用 4-8 个
- **数据增强**: 训练时可选，增加鲁棒性

## 测试

运行 Data 模块测试：

```bash
python tests/test_data_module.py
```

测试覆盖:
- ✅ 基础映射库构建和查询
- ✅ 采样策略（逆向扰动、时间参数、流路径）
- ✅ Flow 采样器（单样本和批量采样）
- ✅ 数据集创建、保存和加载
- ✅ DataLoader 创建和数据增强

## API 参考

完整的 API 文档请参考各模块的 docstring：

```python
from mfik import data
help(data.BaseMapping)
help(data.FlowSampler)
help(data.IKFlowDataset)
help(data.create_dataloader)
```

## 常见问题

### Q: IK 求解失败怎么办？

A: 可能的原因：
1. 目标位姿超出工作空间
2. IK 初值距离目标较远
3. 数值优化参数需要调整

解决方法：
- 使用基础映射库提供好的初值
- 增加 IK 迭代次数 (`max_iter=1000`)
- 调整容差 (`tolerance=1e-5`)

### Q: 数据生成速度慢怎么办？

A: 优化建议：
1. 使用 GPU (`--device cuda`)
2. 增大批量大小 (`--batch-size 128`)
3. 复用基础映射库 (`--load-base-mapping`)
4. 减少基础映射库样本数

### Q: 如何验证数据质量？

A: 验证方法：
```python
# 加载数据集
dataset = IKFlowDataset(data_path='data.pt')

# 查看统计信息
stats = dataset.get_statistics()
print(stats)

# 检查精确解的 FK 误差
from mfik.robot.forward_kinematics import ForwardKinematics
fk = ForwardKinematics(chain)
pos_pred, quat_pred = fk.compute(dataset.q_star[:10])
pos_error = torch.norm(pos_pred - dataset.target_pos[:10], dim=1)
print(f'FK 位置误差: {pos_error.mean().item():.6f} m')
```

## 更新日志

- **2024-12**: 初始版本，包含所有核心功能
- **2024-12**: 添加批量采样功能和数据生成脚本
- **2024-12**: 增强测试覆盖率和文档
