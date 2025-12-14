# MeanFlow IK

基于 MeanFlow 的机器人逆运动学求解器

## 快速开始

### 1. 配置环境

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 生成训练数据

```bash
# 为 Panda 机器人生成训练数据（100 万个样本）
python scripts/generate_data.py \
    --robot panda \
    --n-samples 1_000_000 \
    --output data/panda_train.pt \
    --device cuda

# 生成验证数据（10 万个样本）
python scripts/generate_data.py \
    --robot panda \
    --n-samples 100_000 \
    --output data/panda_val.pt \
    --device cuda

# 支持的机器人: panda, ur10
# 也可以使用自定义 URDF 文件
python scripts/generate_data.py \
    --robot panda \
    --urdf-path /path/to/your/robot.urdf \
    --n-samples 10000 \
    --output data/custom_train.pt
```

### 3. 训练模型

```bash
# 训练 Panda 机器人的 IK 模型
python scripts/train.py \
    --robot panda \
    --data_path data/panda_train.pt \
    --output_dir outputs/panda_model \
    --num_epochs 100 \
    --batch_size 128

# 查看训练参数
python scripts/train.py --help
```

### 4. 评估模型

```bash
# 静态姿态测试
python scripts/eval/test_static_poses.py \
    --checkpoint outputs/panda_model/checkpoints/best_model.pth \
    --urdf robots/panda_arm.urdf \
    --n-samples 1000 \
    --device cuda

# 轨迹跟踪测试
python scripts/eval/test_trajectories.py \
    --checkpoint outputs/panda_model/checkpoints/best_model.pth \
    --urdf robots/panda_arm.urdf \
    --device cuda

# 与数值 IK 方法对比
python scripts/eval/compare_numerical_ik.py \
    --checkpoint outputs/panda_model/checkpoints/best_model.pth \
    --urdf robots/panda_arm.urdf \
    --n-samples 1000 \
    --output-dir results/comparison \
    --device cuda
```

## 完整示例（一键运行）

```bash
# 1. 配置环境
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. 生成数据
python scripts/generate_data.py --robot panda --n-samples 10000 --output data/panda_train.pt
python scripts/generate_data.py --robot panda --n-samples 1000 --output data/panda_val.pt

# 3. 训练模型
python scripts/train.py \
    --robot panda \
    --data_path data/panda_train.pt \
    --output_dir outputs/panda \
    --num_epochs 100

# 4. 评估模型
python scripts/eval/compare_numerical_ik.py \
    --checkpoint outputs/panda/checkpoints/best_model.pth \
    --urdf robots/panda_arm.urdf \
    --n-samples 1000
```

## 项目结构

```
mfik/
├── data/           # 数据生成和加载模块
├── model/          # 神经网络模型
│   ├── v1/         # MeanFlow 第一版
│   └── v2/         # 未来版本
├── train/          # 训练模块
├── eval/           # 评估和推理模块
├── robot/          # 机器人运动学模块
│   ├── urdf.py                 # URDF 解析
│   ├── forward_kinematics.py   # 正运动学
│   └── inverse_kinematics.py   # 数值逆运动学
└── scripts/        # 可执行脚本
    ├── generate_data.py        # 数据生成
    ├── train.py                # 模型训练
    └── eval/                   # 评估脚本
        ├── test_static_poses.py
        ├── test_trajectories.py
        └── compare_numerical_ik.py
```

## 注意事项

- **GPU 加速**: 如果没有 GPU，将所有命令中的 `--device cuda` 改为 `--device cpu`
- **数据量**: 建议训练集至少 10000 个样本，验证集 1000 个样本
- **训练时间**: 取决于硬件配置，GPU 训练通常需要 10-30 分钟
- **查看帮助**: 使用 `python scripts/xxx.py --help` 查看每个脚本的详细参数

## 支持的机器人

- **Panda**: Franka Emika Panda 7-DOF 机械臂
- **UR10**: Universal Robots UR10 6-DOF 机械臂
- **自定义**: 支持任意 URDF 格式的机器人模型

## 依赖项

主要依赖:
- PyTorch >= 1.13.0
- NumPy >= 1.20.0
- NetworkX >= 2.5
- lxml >= 4.6.0
- scipy >= 1.7.0

详见 `requirements.txt`

## License

MIT License
