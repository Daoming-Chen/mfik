# MeanFlow IK 求解器规格

## ADDED Requirements

### Requirement: 单步 IK 推理接口
系统 SHALL 提供单步 IK 推理接口，输入目标位姿和参考关节角度，输出精确关节解。

#### Scenario: 基本 IK 求解
- **WHEN** 用户输入目标位姿 $x_{target} \in SE(3)$（7D：位置 + 四元数）和参考关节角度 $q_{ref} \in \mathbb{R}^n$
- **THEN** 系统在单次前向传播内输出关节解 $q_{pred}$
- **AND** 满足 $FK(q_{pred}) \approx x_{target}$，位置误差 < 5mm，姿态误差 < 5°
- **AND** 推理延迟 < 1ms（RTX 4090, Batch Size=1）

#### Scenario: 批量 IK 推理
- **WHEN** 用户输入形状为 `[batch_size, 7]` 的目标位姿张量和 `[batch_size, n_joints]` 的参考关节张量
- **THEN** 系统返回 `[batch_size, n_joints]` 的关节解张量
- **AND** 支持 GPU 并行计算

#### Scenario: 处理不可达位姿
- **WHEN** 用户输入的目标位姿超出机械臂工作空间
- **THEN** 系统返回最接近的可达解或抛出 `UnreachableTargetError` 异常
- **AND** 可选地返回置信度分数（0-1）指示解的可靠性

### Requirement: 连续轨迹跟踪模式
系统 SHALL 支持连续轨迹跟踪，利用上一帧的解作为当前帧的参考输入。

#### Scenario: 轨迹跟踪初始化
- **WHEN** 用户开始跟踪新轨迹的第一个位姿
- **THEN** 系统使用零配置或指定的初始关节角度作为 $q_{ref}$

#### Scenario: 轨迹跟踪更新
- **WHEN** 用户输入轨迹的第 $t$ 帧位姿 $x_t$
- **THEN** 系统使用第 $t-1$ 帧的输出 $q_{t-1}$ 作为参考
- **AND** 输出关节解 $q_t$ 满足平滑性约束：$\|q_t - q_{t-1}\|_\infty < 0.1 \, \text{rad}$

#### Scenario: 轨迹跟踪中的跳变检测
- **WHEN** 连续两帧之间的关节差异 $\|q_t - q_{t-1}\|$ 超过阈值
- **THEN** 系统可选地触发警告或重新初始化参考

### Requirement: 关节限位后处理
系统 SHALL 对推理输出进行关节限位裁剪，确保解的物理合法性。

#### Scenario: 裁剪超限关节
- **WHEN** 网络输出的关节角度超出 URDF 定义的限位
- **THEN** 系统将越界关节裁剪到边界值
- **AND** 记录警告日志

#### Scenario: 验证裁剪后的解
- **WHEN** 裁剪后的关节解 $q_{clipped}$ 可能不再满足 $FK(q_{clipped}) \approx x_{target}$
- **THEN** 系统重新计算实际位姿误差并返回给用户

### Requirement: 奇异构型处理
系统 SHALL 检测并处理奇异构型（雅可比矩阵秩亏）。

#### Scenario: 检测奇异点
- **WHEN** 当前关节配置接近奇异点（雅可比条件数 > $10^4$）
- **THEN** 系统可选地降低精度要求或 fallback 到数值优化

#### Scenario: 奇异点处的精度降级
- **WHEN** 目标位姿导致解接近奇异构型
- **THEN** 系统在推理前发出警告，允许用户调整误差阈值

### Requirement: 多解处理策略
系统 SHALL 提供选项，在多解场景下根据策略选择解。

#### Scenario: 选择最近解
- **WHEN** 用户指定 `strategy="closest"`
- **THEN** 系统输出与参考 $q_{ref}$ 欧几里得距离最小的解

#### Scenario: 选择最小运动解
- **WHEN** 用户指定 `strategy="min_motion"`
- **THEN** 系统输出 $\|q_{pred} - q_{ref}\|_2$ 最小的解（等价于 `closest` 在当前设计中）

#### Scenario: 避免关节限位边界
- **WHEN** 用户指定 `strategy="avoid_limits"`
- **THEN** 系统优先输出远离关节限位边界的解

### Requirement: 模型加载与管理
系统 SHALL 支持加载预训练模型权重并管理多个机器人模型。

#### Scenario: 从检查点加载模型
- **WHEN** 用户调用 `IKSolver.from_checkpoint("panda_model.pth")`
- **THEN** 系统加载网络权重、优化器状态和训练配置

#### Scenario: 模型设备管理
- **WHEN** 用户指定 `device="cuda:0"`
- **THEN** 系统将模型参数和推理数据自动移动到指定 GPU

#### Scenario: 多模型并行推理
- **WHEN** 用户需要同时使用 Panda 和 UR10 模型
- **THEN** 系统支持实例化多个 IKSolver，独立管理各自的权重

### Requirement: 推理性能监控
系统 SHALL 提供推理延迟和精度的实时监控工具。

#### Scenario: 记录推理延迟
- **WHEN** 用户启用 `profiling=True`
- **THEN** 系统记录每次推理的耗时（预处理、网络前向、后处理）

#### Scenario: 统计位姿误差
- **WHEN** 用户在测试集上批量推理
- **THEN** 系统自动计算并返回位置误差和姿态误差的均值、中位数、95 分位数

#### Scenario: 生成性能报告
- **WHEN** 用户调用 `solver.benchmark(test_poses)`
- **THEN** 系统生成 Markdown 报告，包含精度分布图和延迟直方图

