# Scripts 目录

本目录包含用于数据生成、训练和评估的脚本工具。

## 可用脚本

### generate_data.py - 数据生成脚本

生成 MeanFlow IK 训练数据集。

**快速开始**:
```bash
# 生成小规模测试数据集 (100 样本)
python scripts/generate_data.py \
    --robot panda \
    --n-samples 100 \
    --base-mapping-samples 500 \
    --output data/test_panda_100.pt

# 生成训练数据集 (10k 样本)
python scripts/generate_data.py \
    --robot panda \
    --n-samples 10000 \
    --output data/panda_10k.pt \
    --device cuda

# 生成大规模数据集 (100k 样本)
python scripts/generate_data.py \
    --robot panda \
    --n-samples 100000 \
    --base-mapping-samples 200000 \
    --output data/panda_100k.pt \
    --save-base-mapping data/panda_base_mapping.npz \
    --device cuda \
    --verbose
```

**使用已有的基础映射库**:
```bash
# 第一次生成：保存基础映射库
python scripts/generate_data.py \
    --robot panda \
    --n-samples 50000 \
    --output data/panda_50k_part1.pt \
    --save-base-mapping data/panda_base_mapping.npz

# 后续生成：加载基础映射库（加速数据生成）
python scripts/generate_data.py \
    --robot panda \
    --n-samples 50000 \
    --output data/panda_50k_part2.pt \
    --load-base-mapping data/panda_base_mapping.npz
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robot` | str | panda | 机器人模型 (panda, ur10) |
| `--urdf-path` | str | None | 自定义 URDF 文件路径 |
| `--n-samples` | int | 10000 | 生成的样本数量 |
| `--base-mapping-samples` | int | 100000 | 基础映射库样本数量 |
| `--noise-std` | float | 0.1 | 扰动噪声标准差 (rad) |
| `--time-distribution` | str | beta | 时间参数分布 (uniform, beta, logit_normal) |
| `--batch-size` | int | 32 | IK 求解批量大小 |
| `--output` | str | 必需 | 输出文件路径 (.pt 或 .npz) |
| `--save-base-mapping` | str | None | 保存基础映射库路径 |
| `--load-base-mapping` | str | None | 加载基础映射库路径 |
| `--device` | str | cpu | 计算设备 (cpu, cuda) |
| `--seed` | int | 42 | 随机种子 |
| `--verbose` | flag | False | 显示详细输出 |

**性能建议**:

1. **使用 GPU**: 对于大规模数据生成（> 10k 样本），强烈建议使用 GPU
   ```bash
   python scripts/generate_data.py --device cuda ...
   ```

2. **复用基础映射库**: 生成多个数据集时，保存并复用基础映射库
   ```bash
   # 第一次保存
   --save-base-mapping data/base_mapping.npz
   # 后续加载
   --load-base-mapping data/base_mapping.npz
   ```

3. **调整批量大小**:
   - CPU: 32-64
   - GPU: 128-256

4. **文件格式选择**:
   - 推荐使用 `.pt` 格式（PyTorch 原生，加载速度快）
   - 如需兼容性，使用 `.npz` 格式（NumPy 压缩）

**预期数据规模**:

| 样本数量 | 文件大小 (Panda, 7-DOF) | 生成时间 (估计) |
|---------|------------------------|----------------|
| 1k      | ~0.2 MB                | < 1 分钟        |
| 10k     | ~2 MB                  | 2-5 分钟        |
| 100k    | ~20 MB                 | 20-40 分钟      |
| 1M      | ~200 MB                | 3-6 小时        |

*注：生成时间取决于硬件配置和 IK 求解复杂度*

## 示例工作流

### 1. 开发阶段 - 快速验证

```bash
# 生成小规模测试数据
python scripts/generate_data.py \
    --robot panda \
    --n-samples 1000 \
    --base-mapping-samples 5000 \
    --output data/dev_test.pt \
    --device cpu

# 验证数据
python -c "
from mfik.data import IKFlowDataset
ds = IKFlowDataset(data_path='data/dev_test.pt')
print(f'Dataset size: {len(ds)}')
stats = ds.get_statistics()
print(f'Mean r: {stats[\"r_mean\"]:.4f}')
print(f'Mean t: {stats[\"t_mean\"]:.4f}')
"
```

### 2. 训练阶段 - 生成训练集

```bash
# 生成训练数据集（推荐 100k - 1M 样本）
python scripts/generate_data.py \
    --robot panda \
    --n-samples 100000 \
    --base-mapping-samples 200000 \
    --output data/panda_train_100k.pt \
    --save-base-mapping data/panda_base_mapping.npz \
    --device cuda \
    --seed 42 \
    --verbose

# 生成验证数据集
python scripts/generate_data.py \
    --robot panda \
    --n-samples 10000 \
    --load-base-mapping data/panda_base_mapping.npz \
    --output data/panda_val_10k.pt \
    --device cuda \
    --seed 123
```

### 3. 多机器人支持

```bash
# Panda Arm (7-DOF)
python scripts/generate_data.py \
    --robot panda \
    --n-samples 100000 \
    --output data/panda_train.pt \
    --device cuda

# UR10 (6-DOF)
python scripts/generate_data.py \
    --robot ur10 \
    --n-samples 100000 \
    --output data/ur10_train.pt \
    --device cuda
```

## 故障排除

### 问题：IK 求解失败

**错误信息**: `RuntimeError: IK 求解失败`

**可能原因**:
1. 目标位姿超出工作空间
2. IK 初值距离目标较远
3. 数值优化参数需要调整

**解决方法**:
- 增加基础映射库样本数 (`--base-mapping-samples 200000`)
- 调整扰动噪声 (`--noise-std 0.05`)
- 检查机器人 URDF 文件是否正确

### 问题：生成速度慢

**优化方法**:
1. 使用 GPU: `--device cuda`
2. 增大批量: `--batch-size 128`
3. 复用基础映射库: `--load-base-mapping`

### 问题：内存不足

**解决方法**:
1. 减少批量大小: `--batch-size 16`
2. 减少基础映射库样本: `--base-mapping-samples 50000`
3. 分批生成数据，然后合并

## 未来计划

- [ ] `train.py` - 训练脚本（训练 MeanFlow 网络）
- [ ] `evaluate.py` - 评估脚本（评估模型精度和速度）
- [ ] `visualize.py` - 可视化脚本（工作空间热力图、轨迹跟踪）
- [ ] `benchmark.py` - 性能基准测试脚本
