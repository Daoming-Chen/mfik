# Robot 模块

机器人运动学模块，提供URDF解析、正运动学和逆运动学功能。

## 功能特性

### 1. URDF解析 (`urdf.py`)
- ✅ 解析URDF文件，提取运动学链信息
- ✅ 支持关节限位检查和裁剪
- ✅ 周期性关节的最短弧处理
- ✅ 角度归一化和插值

### 2. 正运动学 (`forward_kinematics.py`)
- ✅ 基于PyTorch的可微分FK计算
- ✅ 支持批量计算
- ✅ CPU和GPU兼容
- ✅ 自动微分支持（可计算雅可比）
- ✅ 中间链接位姿计算

### 3. 逆运动学 (`inverse_kinematics.py`)
- ✅ 雅可比伪逆迭代法
- ✅ 阻尼最小二乘法(LMA)
- ✅ 批量IK求解
- ✅ 数值稳定的梯度计算

## 安装依赖

```bash
pip install torch numpy networkx lxml -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速开始

### 解析URDF
```python
from mfik.robot import parse_urdf

# 解析URDF文件
chain = parse_urdf('robots/panda_arm.urdf')
print(f"关节数量: {chain.n_joints}")
chain.print_tree()
```

### 正运动学
```python
import torch
from mfik.robot import ForwardKinematics, parse_urdf

# 初始化
chain = parse_urdf('robots/panda_arm.urdf')
fk = ForwardKinematics(chain, device='cpu')

# 计算FK
q = torch.zeros(chain.n_joints)
position, quaternion = fk.compute(q)
print(f"位置: {position}")
print(f"姿态(四元数): {quaternion}")

# 批量计算
q_batch = torch.randn(10, chain.n_joints)
positions, quaternions = fk.compute(q_batch)
```

### 逆运动学
```python
import torch
from mfik.robot import InverseKinematics, ForwardKinematics, parse_urdf

# 初始化
chain = parse_urdf('robots/panda_arm.urdf')
fk = ForwardKinematics(chain)
ik = InverseKinematics(chain)

# 生成目标位置
q_target = torch.randn(chain.n_joints) * 0.3
target_pos, _ = fk.compute(q_target)

# 求解IK
q_solved = ik.solve(target_pos, method='lma', max_iter=200)
print(f"求解的关节角度: {q_solved}")

# 验证
solved_pos, _ = fk.compute(q_solved)
error = torch.norm(solved_pos - target_pos)
print(f"位置误差: {error.item()*1000:.2f}mm")
```

## 测试

运行单元测试:
```bash
python -m pytest tests/test_robot_module.py -v
```

测试覆盖:
- ✅ URDF解析 (Panda Arm, UR10)
- ✅ 正运动学计算 (单配置、批量、梯度)
- ✅ 逆运动学求解 (Jacobian方法、LMA方法、批量)
- ✅ 关节工具函数 (限位、角度处理)
- ✅ 集成测试 (FK-IK一致性、轨迹跟踪)

**测试结果: 19/19 通过 ✅**

## 性能特点

### 正运动学
- 单次FK: ~0.1ms (CPU)
- 批量FK (100个配置): ~2ms (CPU)
- 支持GPU加速

### 逆运动学
- LMA方法: 精度可达5cm以内 (200次迭代)
- Jacobian方法: 精度约10-15cm (200次迭代)
- 批量求解支持

## API文档

### 主要类

#### `KinematicChain`
运动学链数据结构,包含关节、链接和限位信息。

#### `ForwardKinematics`
正运动学计算器,支持批量和可微分计算。

#### `InverseKinematics`
逆运动学求解器,支持多种数值方法。

### 主要函数

- `parse_urdf(urdf_path)`: 解析URDF文件
- `get_kinematic_chain()`: 获取运动学链
- `check_joint_limits()`: 检查关节限位
- `clip_to_limits()`: 裁剪到限位范围
- `shortest_angular_distance()`: 计算最短角度差
- `normalize_angles()`: 角度归一化
- `interpolate_angles()`: 角度插值

## 实现细节

### 正运动学
- 使用Rodrigues公式计算旋转变换
- 累积变换矩阵计算末端位姿
- 旋转矩阵到四元数转换(Shepperd方法)

### 逆运动学
- 数值微分计算雅可比矩阵
- 伪逆法或阻尼最小二乘法更新关节角度
- 关节限位约束处理

## 支持的机器人

测试通过的机器人:
- ✅ Franka Panda (7-DoF)
- ✅ UR10 (6-DoF)

理论上支持所有标准URDF格式的串联机器人。

## 局限性

1. 仅支持串联机器人(不支持闭链机构)
2. IK求解为数值方法,精度有限(厘米级)
3. 不包含碰撞检测
4. 不处理奇异点附近的特殊情况

## 下一步

本模块为后续的学习型IK求解器(MeanFlow)提供基础:
- 数值IK用于生成训练数据
- FK用于计算目标位姿
- 运动学链用于定义问题空间

## 贡献者

实现日期: 2025-12-08

