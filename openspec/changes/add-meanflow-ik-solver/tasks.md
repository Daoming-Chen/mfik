# 实施任务清单

## 1. Robot 模块：URDF 解析与运动学
- [x] 1.1 创建模块结构 `mfik/robot/` 目录和 `__init__.py`
- [x] 1.2 重构 `mfik/urdf.py` → `mfik/robot/urdf.py`
  - [x] 1.2.1 精简为纯 URDF 解析功能（移除运动学计算）
  - [x] 1.2.2 提供 `parse_urdf()`, `get_kinematic_chain()` 接口
  - [x] 1.2.3 实现关节限位、周期性处理工具函数
- [x] 1.3 实现 `mfik/robot/forward_kinematics.py`
  - [x] 1.3.1 基于 PyTorch 的 FK 计算（支持批量和自动微分）
  - [x] 1.3.2 支持计算中间链接位姿
  - [x] 1.3.3 CPU 和 GPU 兼容
- [x] 1.4 实现 `mfik/robot/inverse_kinematics.py`
  - [x] 1.4.1 雅可比伪逆迭代法
  - [x] 1.4.2 阻尼最小二乘法（LMA）
  - [x] 1.4.3 批量求解支持
- [x] 1.5 添加单元测试（Panda Arm 和 UR10）
- [ ] 1.6 实现运动学树可视化工具（可选）

## 2. Data 模块：通用数据生成和加载
- [ ] 2.1 创建模块结构 `mfik/data/` 目录和 `__init__.py`
- [ ] 2.2 实现 `mfik/data/base_mapping.py`
  - [ ] 2.2.1 关节空间均匀采样
  - [ ] 2.2.2 批量 FK 计算（调用 `robot.forward_kinematics`）
  - [ ] 2.2.3 KD-Tree 或 FAISS 索引构建
- [ ] 2.3 实现 `mfik/data/sampling.py`
  - [ ] 2.3.1 逆向扰动采样（从 $q^*$ 添加高斯噪声）
  - [ ] 2.3.2 时间参数 $(r, t)$ 采样（logit-normal 分布）
  - [ ] 2.3.3 线性插值流路径生成
- [ ] 2.4 实现 `mfik/data/dataset.py`
  - [ ] 2.4.1 PyTorch Dataset 类（.pt 格式，方便 GPU 快速加载）
  - [ ] 2.4.2 数据预加载和缓存机制
- [ ] 2.5 实现 `mfik/data/loader.py`
  - [ ] 2.5.1 DataLoader 配置（批量大小、多进程）
  - [ ] 2.5.2 数据增强（可选）
- [ ] 2.6 生成小规模数据集（10k 样本）验证管线
- [ ] 2.7 生成大规模训练数据集（Panda: 1M, UR10: 1M）

## 3. Model 模块：版本化神经网络架构
- [ ] 3.1 创建版本化结构 `mfik/model/v1/` 和 `__init__.py`
- [ ] 3.2 实现 `mfik/model/v1/network.py`（MeanFlow ResNet）
  - [ ] 3.2.1 输入特征编码（关节角度 + 目标位姿 + 时间参数）
  - [ ] 3.2.2 位置编码（正弦编码，可选 RFF）
  - [ ] 3.2.3 Input Projection 层（Linear + SiLU）
  - [ ] 3.2.4 Residual Block（Pre-Norm + Linear + SiLU + Residual）
  - [ ] 3.2.5 FiLM 条件注入层
  - [ ] 3.2.6 Output Head（LayerNorm + Linear）
  - [ ] 3.2.7 组装完整网络（6 ResBlocks, 隐层 1024）
- [ ] 3.3 实现 `mfik/model/v1/config.py`（默认超参数）
- [ ] 3.4 创建 `mfik/model/v1/README.md`（架构文档）
- [ ] 3.5 实现模型检查点保存/加载（包含版本元数据）
- [ ] 3.6 预留 `mfik/model/v2/` 目录（未来扩展）
- [ ] 3.7 实现 `mfik/model/utils.py`（版本发现、动态加载）
- [ ] 3.8 验证网络输入/输出维度和参数量

## 4. Train 模块：版本化训练流程
- [ ] 4.1 创建版本化结构 `mfik/train/v1/` 和 `__init__.py`
- [ ] 4.2 实现 `mfik/train/v1/trainer.py`（MeanFlow Trainer）
  - [ ] 4.2.1 MeanFlow 损失计算
  - [ ] 4.2.2 使用 `torch.func.jvp` 实现时间全导数
  - [ ] 4.2.3 stop-gradient 操作
  - [ ] 4.2.4 自适应损失权重 $w = 1/(\|\Delta\|^2 + c)^p$
- [ ] 4.3 实现 `mfik/train/v1/optimizer.py`
  - [ ] 4.3.1 AdamW 优化器配置
  - [ ] 4.3.2 Cosine 学习率调度（含 warmup）
  - [ ] 4.3.3 梯度裁剪
- [ ] 4.4 实现 `mfik/train/v1/config.py`（训练超参数）
- [ ] 4.5 实现训练循环和监控
  - [ ] 4.5.1 批量加载、前向、损失、梯度更新
  - [ ] 4.5.2 TensorBoard 日志（损失、梯度范数）
  - [ ] 4.5.3 检查点保存和恢复
- [ ] 4.6 创建 `mfik/train/v1/README.md`（训练文档）
- [ ] 4.7 预留 `mfik/train/v2/` 目录（未来扩展）
- [ ] 4.8 在 Panda Arm 上训练 v1 模型（验证管线）
- [ ] 4.9 在 UR10 上训练 v1 模型

## 5. Eval 模块：通用推理和评估
- [ ] 5.1 创建模块结构 `mfik/eval/` 目录和 `__init__.py`
- [ ] 5.2 实现 `mfik/eval/inference.py`
  - [ ] 5.2.1 单步 IK 推理接口（版本无关）
  - [ ] 5.2.2 批量推理和 GPU 加速
  - [ ] 5.2.3 动态加载指定版本的模型
  - [ ] 5.2.4 连续轨迹跟踪模式
- [ ] 5.3 实现 `mfik/eval/metrics.py`
  - [ ] 5.3.1 位置误差和姿态误差计算
  - [ ] 5.3.2 成功率统计（误差阈值可配置）
  - [ ] 5.3.3 推理延迟测量
- [ ] 5.4 实现 `mfik/eval/visualization.py`
  - [ ] 5.4.1 精度分布图（箱线图、直方图）
  - [ ] 5.4.2 工作空间热力图
  - [ ] 5.4.3 轨迹跟踪可视化
- [ ] 5.5 实现评估脚本
  - [ ] 5.5.1 静态随机位姿测试（10k 样本）
  - [ ] 5.5.2 连续轨迹测试（直线、圆弧、螺旋线）
  - [ ] 5.5.3 与数值 IK 对比实验
- [ ] 5.6 生成性能基准报告

## 6. 文档与发布
- [ ] 6.1 编写模块 API 文档
  - [ ] 6.1.1 `robot/` 模块文档（URDF、FK、数值 IK）
  - [ ] 6.1.2 `data/` 模块文档（数据生成流程）
  - [ ] 6.1.3 `model/v1/` 模块文档（网络架构）
  - [ ] 6.1.4 `train/v1/` 模块文档（训练流程）
  - [ ] 6.1.5 `eval/` 模块文档（推理和评估）
- [ ] 6.2 准备 Jupyter Notebook 演示
  - [ ] 6.2.1 加载模型和推理示例
  - [ ] 6.2.2 轨迹跟踪演示
  - [ ] 6.2.3 版本对比实验（v1 vs v2）
- [ ] 6.3 编写训练指南
  - [ ] 6.3.1 数据集生成步骤
  - [ ] 6.3.2 超参数配置说明
  - [ ] 6.3.3 多 GPU 训练指南
- [ ] 6.4 导出预训练模型权重（Panda v1, UR10 v1）
- [ ] 6.5 更新 `requirements.txt` 和安装说明
- [ ] 6.6 编写项目 README（包含模块结构说明）

## 7. 测试与验证
- [ ] 7.1 单元测试
  - [ ] 7.1.1 `robot/` 模块测试（FK、数值 IK、URDF 解析）
  - [ ] 7.1.2 `data/` 模块测试（采样、数据加载）
  - [ ] 7.1.3 `model/v1/` 网络测试（输入输出维度）
  - [ ] 7.1.4 `train/v1/` 损失函数测试
  - [ ] 7.1.5 `eval/` 模块测试（metrics 计算）
- [ ] 7.2 集成测试（端到端训练 + 推理）
- [ ] 7.3 回归测试（模型精度不退化）
- [ ] 7.4 性能测试（推理延迟 < 1ms）
- [ ] 7.5 边界条件测试（奇异构型、关节限位）
- [ ] 7.6 版本兼容性测试（v1 模型加载、v2 预留接口）

