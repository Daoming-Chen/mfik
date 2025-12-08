# 变更：添加基于 MeanFlow 的单步 IK 求解器

## Why

逆运动学（IK）求解是机器人控制的核心问题。传统的数据驱动方法在求解 IK 时面临多解冲突、精度瓶颈和迭代求解开销等问题。本项目提出基于 MeanFlow 原理的单步 IK 求解框架，通过学习平均速度场实现真正的单步推理，无需数值积分，显著提升求解精度和推理速度。

## What Changes

- **新增** URDF 机器人模型解析能力，支持从 .urdf 文件加载机器人运动学模型
- **新增** 正运动学（FK）计算模块，支持从关节角度计算末端执行器位姿
- **新增** 高精度训练数据工厂，使用逆向扰动采样策略生成 Flow Matching 数据集
- **新增** 基于 MeanFlow Identity 的神经网络架构和训练流程
- **新增** 单步 IK 推理接口，输入目标位姿和参考关节角度，输出精确关节解
- **新增** 连续轨迹跟踪评估工具，验证模型在实时控制场景下的表现

## Impact

### 受影响的规格 (Affected Specs)
- **新增** `urdf-parser` - URDF 格式解析和机器人运动学建模
- **新增** `ik-solver` - MeanFlow IK 核心求解器
- **新增** `data-factory` - 训练数据生成管线
- **新增** `network-training` - 神经网络训练与优化

### 受影响的代码 (Affected Code)
- `mfik/urdf.py` - 现有的 URDF 解析工具，需要重构并集成到新框架
- `robots/*.urdf` - 机器人模型文件（Panda Arm, UR10），作为测试和训练对象
- **新增** `mfik/` 模块化结构，包含以下子模块：
  - `mfik/robot/` - 机器人运动学模块
    - `robot/urdf.py` - URDF 解析（仅解析运动学链）
    - `robot/forward_kinematics.py` - 基于 PyTorch 的正运动学
    - `robot/inverse_kinematics.py` - 基于 PyTorch 的数值迭代 IK
  - `mfik/data/` - 通用数据生成和加载模块
    - `data/dataset.py` - 数据集接口
    - `data/sampling.py` - 采样策略
    - `data/loader.py` - 数据加载工具
  - `mfik/model/` - 神经网络架构（支持版本迭代）
    - `model/v1/` - 第一版模型架构
    - `model/v2/` - 第二版模型架构（预留）
  - `mfik/train/` - 训练逻辑（支持版本迭代）
    - `train/v1/` - 第一版训练流程
    - `train/v2/` - 第二版训练流程（预留）
  - `mfik/eval/` - 通用评估和推理模块
    - `eval/inference.py` - IK 推理接口
    - `eval/metrics.py` - 评估指标
    - `eval/visualization.py` - 可视化工具

### 技术依赖
- PyTorch (自动微分，JVP 计算)
- NumPy (数值计算)
- NetworkX (运动学树构建)
- Trimesh (几何计算)
- 现有的 `urdf.py` 解析工具

### 性能目标
- **求解精度**：位置误差 < 5mm，姿态误差 < 5°
- **推理延迟**：< 1ms (RTX 4090, Batch Size=1)
- **成功率**：> 95% 在可达工作空间内

