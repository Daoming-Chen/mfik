# Model 和 Train 模块版本化规格

## ADDED Requirements

### Requirement: 版本化目录结构
系统 SHALL 支持多版本的 model 和 train 模块同时存在，采用 `v1/`, `v2/` 子目录组织。

#### Scenario: 创建新版本的模型
- **WHEN** 研究人员需要实验新的网络架构（如从 ResNet 切换到 Transformer）
- **THEN** 系统允许在 `mfik/model/v2/` 创建新模块，而不影响 `v1/`
- **AND** 两个版本可以同时加载和对比

#### Scenario: 版本间独立性
- **WHEN** 用户修改 `model/v2/network.py` 的实现
- **THEN** `model/v1/` 的代码和行为不受影响
- **AND** 已训练的 v1 模型权重仍可正常加载

#### Scenario: 版本命名约定
- **WHEN** 系统识别版本目录
- **THEN** 版本号遵循格式 `v1`, `v2`, `v3`, ...（不使用语义化版本号）
- **AND** 每个版本目录包含独立的 `__init__.py` 声明公共接口

### Requirement: Model 模块版本化
`mfik/model/` 目录 SHALL 包含多个版本的神经网络架构实现。

#### Scenario: v1 模型架构
- **WHEN** 用户导入 `from mfik.model.v1 import MeanFlowNetwork`
- **THEN** 系统加载 v1 版本的网络（ResNet-based, 6 个 ResBlocks, 隐层宽度 1024）
- **AND** 网络接口兼容 `mfik.data` 和 `mfik.eval` 模块

#### Scenario: v2 模型架构（预留）
- **WHEN** 未来实现基于 Transformer 的架构
- **THEN** 代码位于 `mfik/model/v2/transformer.py`
- **AND** 提供相同的接口 `forward(q, target_pose, t)` 用于互换性

#### Scenario: 模型配置文件
- **WHEN** 每个版本包含 `config.py` 定义默认超参数
- **THEN** 用户可通过 `from mfik.model.v1.config import DEFAULT_CONFIG` 获取配置
- **AND** 配置包含：隐层宽度、ResBlock 数量、激活函数类型等

### Requirement: Train 模块版本化
`mfik/train/` 目录 SHALL 包含多个版本的训练逻辑实现。

#### Scenario: v1 训练流程（MeanFlow Loss）
- **WHEN** 用户导入 `from mfik.train.v1 import MeanFlowTrainer`
- **THEN** 系统加载 v1 训练器，使用 MeanFlow Identity 损失
- **AND** 包含 JVP 计算、自适应权重、梯度裁剪等逻辑

#### Scenario: v2 训练流程（预留，如 Diffusion）
- **WHEN** 未来实现基于 Diffusion 的训练方法
- **THEN** 代码位于 `mfik/train/v2/diffusion_trainer.py`
- **AND** 使用不同的损失函数（如 DDPM Loss）

#### Scenario: 训练配置独立性
- **WHEN** v1 使用学习率 `1e-4` 和 Cosine 调度
- **THEN** v2 可以独立配置不同的学习率（如 `5e-5`）和调度器（如 StepLR）
- **AND** 配置通过 `mfik/train/v1/config.py` 和 `mfik/train/v2/config.py` 分别管理

### Requirement: 版本间接口统一
所有版本的 model 和 train 模块 SHALL 遵循统一的接口规范，确保与通用模块（data, eval）兼容。

#### Scenario: Model 接口约定
- **WHEN** 实现任何版本的模型
- **THEN** 必须提供以下方法：
  - `__init__(n_joints, hidden_dim, **kwargs)` - 初始化
  - `forward(q, target_pose, t)` - 前向传播
  - `load_checkpoint(path)` - 加载权重
  - `save_checkpoint(path)` - 保存权重

#### Scenario: Trainer 接口约定
- **WHEN** 实现任何版本的训练器
- **THEN** 必须提供以下方法：
  - `__init__(model, dataloader, config)` - 初始化
  - `train_step(batch)` - 单步训练
  - `evaluate(test_loader)` - 评估
  - `save_checkpoint(path)` - 保存训练状态

#### Scenario: 通用模块调用版本化模块
- **WHEN** `mfik.eval.inference` 需要加载模型
- **THEN** 用户通过 `inference.load_model("path/to/checkpoint", version="v1")` 指定版本
- **AND** 系统自动导入对应版本的模型类

### Requirement: 版本选择和切换
系统 SHALL 提供工具函数用于版本发现和动态加载。

#### Scenario: 列举可用版本
- **WHEN** 用户调用 `mfik.model.list_versions()`
- **THEN** 系统返回 `["v1", "v2"]`（扫描 `model/` 目录下的 `v*` 子目录）

#### Scenario: 动态加载指定版本
- **WHEN** 用户调用 `mfik.model.load_version("v2")`
- **THEN** 系统动态导入 `mfik.model.v2` 模块
- **AND** 返回该版本的公共接口（网络类、配置等）

#### Scenario: 默认版本回退
- **WHEN** 用户未指定版本号
- **THEN** 系统默认使用最新稳定版本（通过 `mfik/model/DEFAULT_VERSION = "v1"` 配置）

### Requirement: 版本文档和元数据
每个版本目录 SHALL 包含 `README.md` 描述该版本的特性和变更。

#### Scenario: 版本 README 内容
- **WHEN** 用户查看 `mfik/model/v1/README.md`
- **THEN** 文件包含以下信息：
  - 架构概述（ResNet-based, 6 ResBlocks）
  - 关键超参数（隐层宽度 1024, SiLU 激活）
  - 性能指标（精度、延迟）
  - 变更日志（相对于前一版本的改进）

#### Scenario: 版本元数据
- **WHEN** 保存模型检查点
- **THEN** 检查点包含版本元数据：
  - `version: "v1"`
  - `architecture: "ResNet"`
  - `created_at: "2025-12-08"`
- **AND** 加载时验证版本匹配


