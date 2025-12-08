# 训练数据生成工厂规格

## ADDED Requirements

### Requirement: 基础映射库构建
系统 SHALL 构建基础映射库 $\mathcal{D}_{base}$，通过关节空间采样和 FK 计算生成位姿-关节对。

#### Scenario: 均匀采样关节空间
- **WHEN** 用户指定每个关节的采样点数 $N_{samples}$
- **THEN** 系统在关节限位范围内生成 $N_{samples}^{n_{joints}}$ 个关节配置
- **AND** 对于 7-DOF 机械臂和 $N_{samples}=10$，生成约 $10^7$ 个样本

#### Scenario: 计算映射库的 FK
- **WHEN** 基础映射库的关节配置生成完成
- **THEN** 系统批量计算 FK，生成位姿库 $\{(q_i, x_i)\}$
- **AND** 使用 GPU 加速，处理速度 > 10k samples/s

#### Scenario: 过滤不可达配置
- **WHEN** 某些关节配置导致碰撞或违反物理约束
- **THEN** 系统从映射库中移除这些样本（MVP 阶段跳过碰撞检测）

### Requirement: 精确解生成
系统 SHALL 使用数值优化器从映射库中检索种子解并精修为高精度解。

#### Scenario: 检索 K 近邻种子解
- **WHEN** 用户指定目标位姿 $x_{target}$
- **THEN** 系统在基础映射库中检索 K=5 个最接近的位姿对应的关节配置
- **AND** 使用 KD-Tree 或 FAISS 索引加速检索（位置空间）

#### Scenario: 数值优化精修
- **WHEN** 系统获得种子解 $q_{seed}$
- **THEN** 使用阻尼最小二乘法（LMA）优化 $\min_q \|FK(q) \ominus x_{target}\|^2$
- **AND** 优化收敛后位姿误差 < $10^{-6}$

#### Scenario: 处理优化失败
- **WHEN** 所有 K 个种子解均无法收敛到精度阈值
- **THEN** 系统丢弃该目标位姿，不生成训练样本

#### Scenario: GPU 加速批量优化
- **WHEN** 需要生成大规模数据集（> 1M 样本）
- **THEN** 系统并行优化多个目标位姿，利用 GPU 计算雅可比矩阵和梯度

### Requirement: 逆向扰动采样
系统 SHALL 实现逆向扰动采样策略，从精确解生成训练数据。

#### Scenario: 从精确解添加高斯噪声
- **WHEN** 系统从映射库随机抽取精确解 $(q^*, x_{target})$
- **THEN** 生成输入关节配置 $q_{input} = q^* + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$
- **AND** 噪声标准差 $\sigma$ 可配置（默认 0.1 rad）

#### Scenario: 处理旋转关节周期性
- **WHEN** 添加噪声后关节角度超出 $[-\pi, \pi]$ 范围
- **THEN** 系统将角度归一化到主值区间

#### Scenario: 确保输入在关节限位内
- **WHEN** $q_{input}$ 中有关节超出限位
- **THEN** 系统重新采样或裁剪到边界值

### Requirement: 流路径生成
系统 SHALL 生成从输入配置到精确解的线性插值流路径。

#### Scenario: 计算线性插值路径
- **WHEN** 给定 $q_{input}$ 和 $q^*$
- **THEN** 系统生成 $q_t = (1-t) q_{input} + t \cdot q^*$，$t \in [0, 1]$
- **AND** 对旋转关节使用最短弧插值

#### Scenario: 采样流路径上的点
- **WHEN** 训练需要在流路径上采样中间点
- **THEN** 系统从分布中采样 $t \sim \text{Uniform}(0, 1)$ 或 $\text{Beta}(2, 5)$
- **AND** 计算对应的 $q_t$

#### Scenario: 计算条件速度
- **WHEN** 流路径定义完成
- **THEN** 系统计算 $v_t = q^* - q_{input}$（对旋转关节使用最短弧差分）

### Requirement: 时间参数采样
系统 SHALL 采样时间参数 $(r, t)$ 用于 MeanFlow 损失计算。

#### Scenario: 采样 logit-normal 分布
- **WHEN** 训练循环需要 $(r, t)$ 对
- **THEN** 系统从 logit-normal 分布采样，确保 $0 \leq r < t \leq 1$
- **AND** 参数可配置（均值、方差）以控制分布形状

#### Scenario: 处理边界情况
- **WHEN** 采样得到 $t - r < \epsilon_{min}$（如 $10^{-3}$）
- **THEN** 系统重新采样或设置 $t = r + \epsilon_{min}$ 避免除零

### Requirement: 数据集存储与加载
系统 SHALL 将生成的数据集保存为磁盘文件，支持高效加载。

#### Scenario: 保存为 HDF5 格式
- **WHEN** 用户指定 `format="hdf5"`
- **THEN** 系统将数据集存储为 HDF5，包含字段：`q_input`, `q_exact`, `x_target`
- **AND** 支持增量写入，避免内存溢出

#### Scenario: 保存为 Parquet 格式
- **WHEN** 用户指定 `format="parquet"`
- **THEN** 系统将数据集存储为 Parquet，支持列式压缩和快速过滤

#### Scenario: 数据集分片
- **WHEN** 数据集大小 > 10GB
- **THEN** 系统自动分片存储（如每 100k 样本一个文件）

#### Scenario: PyTorch DataLoader 集成
- **WHEN** 用户调用 `IKDataset(path="data.hdf5")`
- **THEN** 系统返回 PyTorch Dataset 对象，支持多进程加载和随机采样

### Requirement: 数据质量验证
系统 SHALL 验证生成数据的物理合理性和统计分布。

#### Scenario: 验证 FK 一致性
- **WHEN** 数据生成完成
- **THEN** 系统随机抽取 1000 个样本，验证 $FK(q^*) \approx x_{target}$
- **AND** 所有样本的位姿误差 < $10^{-6}$

#### Scenario: 可视化关节分布
- **WHEN** 用户调用 `visualize_dataset(data)`
- **THEN** 系统生成直方图显示各关节角度的分布
- **AND** 高亮显示接近限位边界的样本比例

#### Scenario: 检测数据偏差
- **WHEN** 工作空间某些区域的样本密度过低（< 10 samples/m³）
- **THEN** 系统发出警告，建议增加该区域的采样

### Requirement: 数据增强策略
系统 SHALL 支持可选的数据增强以提升模型鲁棒性。

#### Scenario: 位姿噪声注入
- **WHEN** 用户启用 `augment_pose=True`
- **THEN** 系统向目标位姿 $x_{target}$ 添加小噪声（位置 ≤ 1mm，姿态 ≤ 0.2°）

#### Scenario: 关节噪声注入
- **WHEN** 用户启用 `augment_joints=True`
- **THEN** 系统向 $q_{input}$ 和 $q^*$ 添加小噪声（≤ 0.01 rad）

#### Scenario: 随机工作空间采样
- **WHEN** 用户指定 `sampling_strategy="workspace_uniform"`
- **THEN** 系统在笛卡尔工作空间（而非关节空间）均匀采样目标位姿

