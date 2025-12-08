# Robot 模块规格：机器人运动学

## ADDED Requirements

### Requirement: URDF 文件解析
`robot.urdf` 模块 SHALL 从 .urdf 文件解析机器人模型，构建运动学树（链接、关节、变换矩阵），但**不实现**运动学计算。

#### Scenario: 解析 Panda Arm 模型
- **WHEN** 用户调用 `urdf.parse_urdf("panda_arm.urdf")`
- **THEN** 系统返回运动学链数据结构，包含：
  - 7 个旋转关节的定义（名称、类型、父子链接、轴向）
  - 关节限位（下界、上界）从 `<limit>` 标签提取
  - 各关节的局部变换矩阵（origin, axis）
- **AND** 返回结构可被 `forward_kinematics` 和 `inverse_kinematics` 模块使用

#### Scenario: 处理不存在的文件
- **WHEN** 用户提供的 URDF 文件路径不存在
- **THEN** 系统抛出 `FileNotFoundError` 异常

#### Scenario: 处理格式错误的 URDF
- **WHEN** URDF 文件缺少必需的 `<robot>` 根节点或关键字段
- **THEN** 系统抛出 `ValueError` 异常并提供错误描述

#### Scenario: 提取运动学链（Serial Chain）
- **WHEN** 用户调用 `urdf.get_kinematic_chain(root_link, end_effector_link)`
- **THEN** 系统返回从根到末端执行器的关节序列
- **AND** 忽略非活动关节（fixed joints）

### Requirement: PyTorch 正运动学计算
`robot.forward_kinematics` 模块 SHALL 实现基于 PyTorch 的可微分正运动学，支持批量计算和自动微分。

#### Scenario: 计算单个配置的 FK
- **WHEN** 用户调用 `fk.compute(q)` 输入 7 维关节角度向量 $q \in \mathbb{R}^7$
- **THEN** 系统返回末端执行器位姿，包含位置 $(x, y, z)$ 和四元数 $(w, x, y, z)$
- **AND** 计算结果与手动链乘变换矩阵的结果误差 < $10^{-6}$

#### Scenario: 批量 FK 计算
- **WHEN** 用户输入形状为 `[batch_size, n_joints]` 的关节角度张量
- **THEN** 系统返回形状为 `[batch_size, 7]` 的位姿张量（3D 位置 + 4D 四元数）
- **AND** 支持 CPU 和 GPU（CUDA Tensor）计算

#### Scenario: 自动微分支持
- **WHEN** 用户对 FK 输出相对于输入关节角度计算梯度
- **THEN** 系统通过 PyTorch 自动微分正确计算雅可比矩阵 $J = \frac{\partial FK(q)}{\partial q}$
- **AND** 梯度与数值差分结果误差 < $10^{-4}$

#### Scenario: 计算中间链接位姿
- **WHEN** 用户指定 `link_name="panda_link5"`
- **THEN** 系统返回指定链接（非末端执行器）的位姿
- **AND** 用于数据验证和可视化

### Requirement: 数值迭代逆运动学
`robot.inverse_kinematics` 模块 SHALL 实现基于 PyTorch 的数值迭代 IK 算法，作为 baseline 和数据生成工具。

#### Scenario: 雅可比伪逆法求解 IK
- **WHEN** 用户调用 `ik.solve(target_pose, q_init, method="jacobian")`
- **THEN** 系统使用雅可比伪逆迭代求解，返回关节解 $q^*$
- **AND** 满足 $\|FK(q^*) - target\_pose\| < \epsilon$（可配置的收敛阈值）
- **AND** 最多迭代 100 次

#### Scenario: 阻尼最小二乘法（LMA）求解 IK
- **WHEN** 用户调用 `ik.solve(target_pose, q_init, method="lma")`
- **THEN** 系统使用阻尼最小二乘法（Levenberg-Marquardt）求解
- **AND** 在奇异构型附近更稳定（相比雅可比伪逆）
- **AND** 精度可达 $10^{-6}$（用于数据生成）

#### Scenario: 处理优化失败
- **WHEN** 数值优化未能在最大迭代次数内收敛
- **THEN** 系统返回 `None` 或抛出 `OptimizationFailureError` 异常
- **AND** 记录当前误差和迭代次数

#### Scenario: 批量 IK 求解
- **WHEN** 用户输入形状为 `[batch_size, 7]` 的目标位姿
- **THEN** 系统并行求解多个 IK 问题（每个独立迭代）
- **AND** 利用 GPU 加速雅可比矩阵计算

### Requirement: 关节限位验证
`robot.urdf` 和 `robot.inverse_kinematics` 模块 SHALL 提供关节限位验证工具。

#### Scenario: 检测超限关节
- **WHEN** 用户调用 `urdf.check_joint_limits(q)` 输入关节角度
- **THEN** 系统返回布尔数组 `[batch_size, n_joints]` 指示每个关节是否在限位内
- **AND** 可选地抛出 `JointLimitViolationError` 异常

#### Scenario: 裁剪超限关节
- **WHEN** 用户调用 `urdf.clip_to_limits(q)`
- **THEN** 系统将越界关节裁剪到 `[lower, upper]` 边界值
- **AND** 返回裁剪后的关节角度

#### Scenario: 处理无限制关节
- **WHEN** 关节类型为 `continuous`（如连续旋转关节）
- **THEN** 系统不对该关节应用限位检查
- **AND** 但会进行周期性归一化（映射到 $[-\pi, \pi]$）

### Requirement: 周期性关节的最短弧差分
`robot.urdf` 模块 SHALL 提供工具函数处理旋转关节的周期性（$\pm \pi$）。

#### Scenario: 计算最短角度差
- **WHEN** 用户调用 `urdf.shortest_angular_distance(q1=3.0, q2=-3.0)`
- **THEN** 系统返回 $\Delta q = 0.283$ 而非 $6.0$（走最短路径）
- **AND** 处理跨越 $\pm \pi$ 边界的情况

#### Scenario: 归一化角度到 $[-\pi, \pi]$
- **WHEN** 用户调用 `urdf.normalize_angles(q)`
- **THEN** 系统将角度映射到 $[-\pi, \pi]$ 区间
- **AND** 保持角度的等价性（相差 $2\pi k$）

#### Scenario: 生成最短弧插值路径
- **WHEN** 用户调用 `urdf.interpolate_angles(q0, q1, t)`
- **THEN** 系统生成 $q_t = q_0 + t \cdot \Delta q_{shortest}$
- **AND** 对于旋转关节，插值路径沿着最短测地线（不绕行整圈）

### Requirement: 运动学树可视化
`robot.urdf` 模块 SHALL 提供可选的运动学树结构可视化工具。

#### Scenario: 打印关节树
- **WHEN** 用户调用 `kinematic_chain.print_tree()`
- **THEN** 系统输出树形结构，显示每个关节的：
  - 父子链接关系
  - 关节类型（revolute, prismatic, continuous, fixed）
  - 关节限位（如果适用）
  - 局部变换（origin xyz rpy）

#### Scenario: 导出为 DOT 格式
- **WHEN** 用户调用 `kinematic_chain.export_graph("tree.dot")`
- **THEN** 系统生成 Graphviz DOT 文件
- **AND** 文件可用 `dot` 命令渲染为图像

