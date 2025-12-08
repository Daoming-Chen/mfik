# 项目上下文

## 项目目的
MeanFlow IK (mfik) 是一个基于深度学习的逆运动学（Inverse Kinematics）求解器，专注于实现高精度、低延迟的单步 IK 推理。项目采用 MeanFlow 原理，通过学习平均速度场实现从任意参考构型到目标构型的最优位移场预测，无需传统 Flow Matching 方法的多步数值积分。

### 核心目标
- **高精度**：位置误差 < 5mm，姿态误差 < 5°，满足工业级控制需求
- **低延迟**：单步推理 < 1ms，支持实时轨迹跟踪（100+ Hz 控制频率）
- **物理可靠**：通过逆向扰动采样生成训练数据，保证解的物理合理性
- **通用性**：支持从 URDF 文件加载任意串联机械臂模型

## 技术栈
- **核心框架**：PyTorch 1.13+（支持 `torch.func.jvp` 进行 JVP 计算）
- **数值计算**：NumPy, SciPy（数值优化）
- **URDF 解析**：lxml, NetworkX（运动学树构建）
- **几何计算**：Trimesh（可选，用于碰撞检测和可视化）
- **数据存储**：HDF5 / Parquet（大规模训练数据）
- **实验跟踪**：TensorBoard / Weights & Biases（训练监控）
- **测试框架**：pytest（单元测试和集成测试）

## 项目约定

### 代码风格
- **Python 风格**：遵循 PEP 8，使用 Black 格式化（行长度 100）
- **类型注解**：所有公共 API 必须包含类型提示（Type Hints）
- **文档字符串**：使用 NumPy/SciPy 风格的 docstring，包含参数、返回值和示例
- **命名约定**：
  - 模块/包：小写下划线（`ik_solver.py`）
  - 类：大驼峰（`MeanFlowNetwork`）
  - 函数/变量：小写下划线（`compute_forward_kinematics`）
  - 常量：大写下划线（`DEFAULT_BATCH_SIZE`）

### 架构模式
- **模块化设计**：核心功能拆分为独立模块（robot, data, network, train, inference）
- **配置管理**：使用 YAML 文件管理训练超参数，支持命令行覆盖
- **日志规范**：使用 Python logging 模块，区分 DEBUG/INFO/WARNING/ERROR 级别
- **错误处理**：自定义异常类（`UnreachableTargetError`, `JointLimitViolationError`）
- **资源管理**：使用上下文管理器（Context Manager）管理 GPU 内存和文件 I/O

### 测试策略
- **单元测试**：覆盖所有核心函数（FK 计算、数据生成、损失函数）
- **集成测试**：端到端测试（数据生成 → 训练 → 推理）
- **回归测试**：使用固定的测试数据集验证模型精度不退化
- **性能测试**：基准测试推理延迟（目标 < 1ms）
- **测试数据**：使用 Panda Arm 和 UR10 作为标准测试机器人

### Git 工作流
- **分支策略**：
  - `main`：稳定发布版本
  - `dev`：开发分支
  - `feature/*`：新功能分支
  - `fix/*`：Bug 修复分支
- **提交规范**：使用 Conventional Commits 格式（`feat:`, `fix:`, `docs:`, `refactor:`, `test:`）
- **PR 要求**：
  - 所有新功能必须包含测试
  - 通过 CI 检查（代码风格、单元测试）
  - 至少一位审阅者批准

## 领域上下文

### 逆运动学问题
逆运动学（IK）是机器人学的核心问题：给定末端执行器的目标位姿 $x_{target} \in SE(3)$（3D 位置 + 3D 姿态），求解关节角度 $q \in \mathbb{R}^n$ 使得正运动学 $FK(q) = x_{target}$。

**核心挑战**：
1. **多解性**：冗余自由度机械臂（如 7-DOF Panda）对应无数个解
2. **奇异点**：雅可比矩阵秩亏导致数值方法不稳定
3. **实时性**：工业控制要求求解延迟 < 1ms

### MeanFlow 原理
传统 Flow Matching 方法通过学习瞬时速度场 $v_t$ 并在推理时积分 $q_1 = q_0 + \int_0^1 v_t dt$，需要多步迭代。MeanFlow 引入**平均速度场** $u(q, 0, 1) = \int_0^1 v_t dt$，使网络能够单步输出完整位移 $q_1 = q_0 + u(q_0, 0, 1)$。

**核心创新**：
- **MeanFlow Identity**：$u = v - (t-r) \frac{du}{dt}$，揭示平均速度与瞬时速度的内在关系
- **逆向扰动采样**：从精确解 $q^*$ 添加噪声生成输入 $q_{input}$，避免多解边界的速度场不连续
- **JVP 高效计算**：通过 `torch.func.jvp` 在单次传播内计算时间全导数

### URDF 格式
Unified Robot Description Format (URDF) 是 ROS 生态的标准机器人模型格式，基于 XML 描述：
- **Link**：刚体连杆，包含视觉/碰撞几何和惯性参数
- **Joint**：关节（revolute 旋转/prismatic 平移/fixed 固定/continuous 连续旋转）
- **Transform**：父子坐标系之间的刚体变换（origin, xyz, rpy）

**项目中的 URDF**：
- `robots/panda_arm.urdf`：Franka Emika Panda（7-DOF 协作机械臂）
- `robots/ur10.urdf`：Universal Robots UR10（6-DOF 工业机械臂）

## 重要约束
1. **精度优先**：模型精度必须满足工业级要求（位置 < 5mm，姿态 < 5°）
2. **延迟限制**：单步推理必须 < 1ms，确保实时控制可行性
3. **物理合法性**：生成的关节配置必须满足关节限位，避免碰撞（MVP 阶段仅限位检查）
4. **单机器人模型**：当前不支持多机器人通用模型（需要独立训练）
5. **串联机构限制**：仅支持串联机械臂，不支持闭链/并联机构
6. **仅运动学**：不处理动力学（力/力矩），仅关注位置/姿态

## 外部依赖
- **URDF 解析后端**：复用项目中现有的 `urdf.py`（基于 lxml + NetworkX）
- **GPU 计算**：训练和推理依赖 NVIDIA GPU（推荐 RTX 4090 或更高）
- **数值优化器**：数据生成阶段使用 SciPy 的 LMA（阻尼最小二乘法）
- **无外部 API**：所有计算在本地完成，无需联网服务

## 性能基准
- **训练时间**：在 RTX 4090 上训练 Panda Arm 模型约 8-12 小时（1M 样本，100k 步）
- **数据生成**：构建 1M 样本数据集约 2-4 小时（取决于数值优化收敛速度）
- **推理延迟**：< 1ms（Batch Size=1），< 10ms（Batch Size=1024）
- **模型大小**：约 50-100 MB（1024 隐层宽度，6 个 ResBlocks）
- **显存占用**：训练约 8-12 GB（Batch Size=4096），推理 < 1 GB

## 项目结构
```
mfik/
├── mfik/                    # 核心包（模块化架构）
│   ├── robot/               # 机器人运动学模块
│   │   ├── __init__.py
│   │   ├── urdf.py          # URDF 解析（仅运动学链构建）
│   │   ├── forward_kinematics.py  # 基于 PyTorch 的 FK
│   │   └── inverse_kinematics.py  # 基于 PyTorch 的数值迭代 IK
│   ├── data/                # 通用数据生成和加载模块
│   │   ├── __init__.py
│   │   ├── base_mapping.py  # 基础映射库构建
│   │   ├── sampling.py      # 采样策略（逆向扰动）
│   │   ├── dataset.py       # PyTorch Dataset
│   │   └── loader.py        # DataLoader 配置
│   ├── model/               # 神经网络架构（支持版本迭代）
│   │   ├── __init__.py
│   │   ├── utils.py         # 版本发现和动态加载
│   │   ├── v1/              # 第一版模型（ResNet-based MeanFlow）
│   │   │   ├── __init__.py
│   │   │   ├── network.py   # 网络定义
│   │   │   ├── config.py    # 默认超参数
│   │   │   └── README.md    # 架构文档
│   │   └── v2/              # 第二版模型（预留，未来扩展）
│   │       └── __init__.py
│   ├── train/               # 训练逻辑（支持版本迭代）
│   │   ├── __init__.py
│   │   ├── v1/              # 第一版训练（MeanFlow Loss）
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py   # 训练器
│   │   │   ├── optimizer.py # 优化器配置
│   │   │   ├── config.py    # 训练超参数
│   │   │   └── README.md    # 训练文档
│   │   └── v2/              # 第二版训练（预留）
│   │       └── __init__.py
│   └── eval/                # 通用评估和推理模块
│       ├── __init__.py
│       ├── inference.py     # IK 推理接口
│       ├── metrics.py       # 评估指标
│       └── visualization.py # 可视化工具
├── robots/                  # URDF 模型文件
│   ├── panda_arm.urdf
│   └── ur10.urdf
├── configs/                 # 训练配置文件（YAML）
│   ├── panda_v1.yaml
│   └── ur10_v1.yaml
├── tests/                   # 测试文件
│   ├── test_robot/          # robot 模块测试
│   │   ├── test_urdf.py
│   │   ├── test_fk.py
│   │   └── test_ik.py
│   ├── test_data/           # data 模块测试
│   ├── test_model/          # model 模块测试
│   ├── test_train/          # train 模块测试
│   └── test_eval/           # eval 模块测试
├── scripts/                 # 脚本工具
│   ├── generate_data.py     # 数据生成脚本
│   ├── train.py             # 训练脚本（支持版本选择）
│   └── evaluate.py          # 评估脚本
├── notebooks/               # Jupyter 演示
│   ├── demo_v1.ipynb        # v1 模型演示
│   └── version_comparison.ipynb  # 版本对比
├── requirements.txt         # Python 依赖
├── README.md                # 项目文档
└── openspec/                # OpenSpec 规格管理
    ├── project.md
    ├── specs/
    └── changes/
```

## 开发优先级
1. **MVP 阶段**（第 1-4 周）：
   - 基础 FK 实现和数据生成管线
   - 简化的 MeanFlow 网络（无 FiLM）
   - Panda Arm 单机器人训练
   
2. **功能完善**（第 5-6 周）：
   - FiLM 条件注入和自适应权重
   - UR10 支持和跨机器人对比
   - 混合精度训练和分布式支持

3. **评估与优化**（第 7-8 周）：
   - 连续轨迹跟踪测试
   - 奇异点处理策略
   - 性能基准报告和文档

## 参考资料
- MeanFlow 论文：[相关理论背景]
- Flow Matching 综述：[生成模型中的流匹配方法]
- URDF 规范：[ROS Wiki URDF Documentation]
- PyTorch JVP 文档：[torch.func.jvp API Reference]
