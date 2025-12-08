# 神经网络训练规格

## ADDED Requirements

### Requirement: 网络架构定义
系统 SHALL 实现宽而浅的 MLP-ResNet 架构，优化推理速度和拟合能力。

#### Scenario: 输入特征编码
- **WHEN** 网络接收输入 $(q_t, r, t, x_{target})$
- **THEN** 系统将关节角度、时间参数和目标位姿编码为统一维度的特征向量
- **AND** 对时间参数应用正弦位置编码（频率范围 $[2^0, 2^{10}]$）

#### Scenario: 位姿特征表示
- **WHEN** 目标位姿 $x_{target}$ 包含位置和四元数
- **THEN** 系统将四元数归一化为单位长度（训练和推理时）
- **AND** 可选地对位置坐标应用随机傅里叶特征（RFF）编码

#### Scenario: Residual Block 结构
- **WHEN** 网络构建 Backbone
- **THEN** 每个 ResBlock 包含：`LayerNorm -> Linear -> SiLU -> Linear -> Residual Connection`
- **AND** 采用 Pre-Norm 结构提升训练稳定性

#### Scenario: FiLM 条件注入
- **WHEN** ResBlock 处理特征
- **THEN** 系统通过 FiLM 层注入时间和目标位姿信息：$h_{out} = \gamma(t, x) \cdot h_{in} + \beta(t, x)$
- **AND** $\gamma, \beta$ 由独立的 MLP 从条件变量生成

#### Scenario: 输出速度限制
- **WHEN** 网络输出速度场 $u$
- **THEN** 系统可选地应用 Tanh Scaling，将输出限制在 $[\Delta q_{min}, \Delta q_{max}]$ 范围内
- **AND** 限制参数从 URDF 关节限位自动推导

### Requirement: MeanFlow 损失计算
系统 SHALL 实现基于 MeanFlow Identity 的训练损失。

#### Scenario: 计算瞬时速度
- **WHEN** 给定流路径样本 $(q_t, q_{input}, q^*)$
- **THEN** 系统计算 $v_t = q^* - q_{input}$（对旋转关节使用最短弧差分）

#### Scenario: 计算时间全导数
- **WHEN** 需要计算 $\frac{d}{dt} u_\theta(q_t, r, t)$
- **THEN** 系统使用 `torch.func.jvp` 计算 $v_t \cdot \nabla_{q_t} u_\theta + \frac{\partial u_\theta}{\partial t}$
- **AND** JVP 计算在单次前向传播内完成，开销 < 16% 额外时间

#### Scenario: 构建目标值
- **WHEN** 计算 MeanFlow 目标
- **THEN** 系统计算 $u_{tgt} = v_t - (t-r) \frac{d}{dt} u_\theta$
- **AND** 对 $u_{tgt}$ 中的 $u_\theta$ 应用 `stop_gradient`，避免高阶梯度

#### Scenario: 计算 L2 损失
- **WHEN** 预测 $u_\theta$ 和目标 $u_{tgt}$ 计算完成
- **THEN** 系统计算 $\mathcal{L} = \|u_\theta - u_{tgt}\|_2^2$

#### Scenario: 自适应损失权重
- **WHEN** 用户启用 `adaptive_weighting=True`
- **THEN** 系统计算 $w = 1/(\|\Delta\|^2 + c)^p$，其中 $p \in [0.5, 1.0]$，$c = 10^{-3}$
- **AND** 最终损失为 $\mathcal{L} = \text{stop\_grad}(w) \cdot \|\Delta\|_2^2$

### Requirement: 优化器配置
系统 SHALL 使用 AdamW 优化器和 Cosine 学习率调度。

#### Scenario: AdamW 参数设置
- **WHEN** 训练开始
- **THEN** 系统初始化 AdamW，学习率 $2 \times 10^{-4}$，betas $(0.9, 0.95)$，weight_decay $10^{-4}$

#### Scenario: Cosine 学习率退火
- **WHEN** 训练进行到第 $t$ 步
- **THEN** 学习率按 Cosine 曲线衰减：$\text{lr}_t = \text{lr}_{min} + \frac{1}{2}(\text{lr}_{max} - \text{lr}_{min})(1 + \cos(\pi t / T))$
- **AND** 前 5k 步线性 warmup

#### Scenario: 梯度裁剪
- **WHEN** 梯度范数 > 1.0
- **THEN** 系统将梯度裁剪到范数 1.0，防止梯度爆炸

### Requirement: 训练循环管理
系统 SHALL 实现完整的训练循环，包含数据加载、前向传播、损失计算和参数更新。

#### Scenario: 批量数据加载
- **WHEN** 训练循环迭代
- **THEN** 系统从 DataLoader 加载批量数据（Batch Size=4096）
- **AND** 支持多进程预加载和 GPU 异步传输

#### Scenario: 混合精度训练
- **WHEN** 用户启用 `mixed_precision=True`
- **THEN** 系统使用 PyTorch AMP（Automatic Mixed Precision）进行 FP16 前向传播和 FP32 梯度累积
- **AND** 训练速度提升 2x，显存占用减少 50%

#### Scenario: 梯度累积
- **WHEN** 批量大小受显存限制
- **THEN** 系统支持梯度累积（如累积 4 个 mini-batch 再更新）
- **AND** 等效批量大小 = mini_batch_size × accumulation_steps

### Requirement: 训练监控
系统 SHALL 实时监控训练指标并记录日志。

#### Scenario: 损失曲线记录
- **WHEN** 每 100 步
- **THEN** 系统记录训练损失、验证损失和学习率到 TensorBoard 或 wandb

#### Scenario: 梯度统计
- **WHEN** 每 500 步
- **THEN** 系统记录梯度范数、参数更新比例（$\|\Delta \theta\| / \|\theta\|$）

#### Scenario: 验证集评估
- **WHEN** 每 5k 步
- **THEN** 系统在验证集上评估位置误差和姿态误差的均值和中位数
- **AND** 如果验证误差未改善持续 5 次，触发早停（可选）

#### Scenario: 训练时间估算
- **WHEN** 训练开始
- **THEN** 系统根据前 100 步的平均耗时估算总训练时间

### Requirement: 检查点管理
系统 SHALL 定期保存和恢复训练检查点。

#### Scenario: 定期保存检查点
- **WHEN** 每 10k 步或每 epoch 结束
- **THEN** 系统保存检查点文件，包含：网络权重、优化器状态、训练步数、最佳验证指标

#### Scenario: 保存最佳模型
- **WHEN** 验证损失达到新低
- **THEN** 系统额外保存 `best_model.pth`，覆盖之前的最佳模型

#### Scenario: 从检查点恢复
- **WHEN** 用户指定 `resume_from="checkpoint_50k.pth"`
- **THEN** 系统加载权重和优化器状态，从第 50k 步继续训练
- **AND** 学习率调度器状态也正确恢复

#### Scenario: 检查点版本管理
- **WHEN** 保存检查点
- **THEN** 系统在文件名中包含步数（如 `model_step50000.pth`）
- **AND** 可选地仅保留最近 N 个检查点，自动删除旧文件

### Requirement: 超参数配置
系统 SHALL 支持通过配置文件或命令行参数指定超参数。

#### Scenario: YAML 配置文件
- **WHEN** 用户提供 `config.yaml`
- **THEN** 系统加载所有训练超参数（学习率、批量大小、网络结构等）

#### Scenario: 命令行覆盖
- **WHEN** 用户通过命令行传递参数（如 `--lr 1e-4`）
- **THEN** 系统覆盖配置文件中的对应字段

#### Scenario: 超参数验证
- **WHEN** 加载配置
- **THEN** 系统验证所有必需字段存在，且值在合理范围内
- **AND** 不合理的配置（如 batch_size=0）触发异常

### Requirement: 分布式训练支持
系统 SHALL 支持多 GPU 分布式训练（可选功能）。

#### Scenario: DataParallel 模式
- **WHEN** 用户指定多个 GPU（如 `--gpus 0,1,2,3`）
- **THEN** 系统使用 PyTorch DataParallel 在多 GPU 上并行前向传播
- **AND** 批量大小自动在 GPU 间分配

#### Scenario: DistributedDataParallel 模式
- **WHEN** 用户启用 `--distributed`
- **THEN** 系统使用 PyTorch DDP，每个进程管理一个 GPU
- **AND** 梯度通过 all-reduce 同步

#### Scenario: 单机多卡性能
- **WHEN** 在 4×RTX 4090 上训练
- **THEN** 训练吞吐量接近单卡的 4 倍（考虑通信开销）

### Requirement: 可复现性保证
系统 SHALL 支持固定随机种子以确保实验可复现。

#### Scenario: 设置全局种子
- **WHEN** 用户指定 `seed=42`
- **THEN** 系统固定 PyTorch、NumPy、Python random 的随机种子

#### Scenario: 确定性算法
- **WHEN** 用户启用 `deterministic=True`
- **THEN** 系统强制 PyTorch 使用确定性算法（可能牺牲速度）
- **AND** CUDA 卷积算法固定为确定性模式

#### Scenario: 可复现性验证
- **WHEN** 使用相同种子和配置训练两次
- **THEN** 两次训练的损失曲线和最终模型权重完全一致（浮点误差内）

