# Eval Module - Inference and Evaluation

通用的 IK 推理和评估工具，独立于具体模型版本。

## 模块结构

```
mfik/eval/
├── __init__.py           # 模块初始化
├── inference.py          # IK 推理接口
├── metrics.py            # 评估指标
└── visualization.py      # 可视化工具
```

## 核心功能

### 1. IK Solver (`inference.py`)

版本无关的 IK 求解器，支持单步推理和批量处理。

#### 基本用法

```python
from mfik.eval.inference import IKSolver
import torch

# 从检查点加载求解器
solver = IKSolver.from_checkpoint(
    checkpoint_path="checkpoints/panda_v1.pth",
    urdf_path="robots/panda_arm.urdf",
    device="cuda",
)

# 单步 IK 推理
target_pose = torch.tensor([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])  # [pos, quat]
q_ref = torch.zeros(7)  # 参考关节角度

q_solution = solver.solve(target_pose, q_ref)
print(f"Solution: {q_solution}")
```

#### 批量推理

```python
# 批量处理多个位姿
batch_size = 100
target_poses = torch.randn(batch_size, 7)  # [batch, 7]
q_refs = torch.zeros(batch_size, 7)

q_solutions = solver.solve(target_poses, q_refs)
print(f"Solutions shape: {q_solutions.shape}")  # [100, 7]
```

#### 轨迹跟踪模式

```python
# 连续轨迹求解
trajectory = torch.randn(200, 7)  # 200 帧位姿
q_init = torch.zeros(7)  # 初始关节配置

q_trajectory, metadata = solver.solve_trajectory(
    trajectory,
    q_init=q_init,
    check_smoothness=True,
    smoothness_threshold=0.1,  # rad
)

print(f"Smoothness violations: {len(metadata['smoothness_violations'])}")
print(f"Avg time per frame: {metadata['avg_time_per_frame'] * 1000:.2f} ms")
```

#### 性能基准测试

```python
# 评估推理性能
test_poses = torch.randn(1000, 7)
results = solver.benchmark(test_poses, warmup_steps=10)

print(f"Avg latency: {results['avg_latency'] * 1000:.3f} ms")
print(f"Throughput: {results['throughput']:.1f} inferences/sec")
print(f"Position error (95%): {results['pos_error_95'] * 1000:.2f} mm")
```

#### 高级选项

```python
# 带置信度评分的推理
q_solution, confidence = solver.solve(
    target_pose,
    q_ref,
    return_confidence=True,
)
print(f"Confidence: {confidence:.3f}")

# 关节限位处理
q_solution = solver.solve(
    target_pose,
    q_ref,
    clip_joints=True,  # 裁剪超限关节
)
```

### 2. 评估指标 (`metrics.py`)

#### 位姿误差计算

```python
from mfik.eval.metrics import compute_pose_error

# 计算预测关节的位姿误差
pos_errors, rot_errors = compute_pose_error(
    q_pred=q_solutions,
    target_pose=target_poses,
    fk=solver.fk,
)

print(f"Position error: {pos_errors.mean().item() * 1000:.2f} mm")
print(f"Rotation error: {rot_errors.mean().item() * 180 / np.pi:.2f}°")
```

#### 成功率统计

```python
from mfik.eval.metrics import compute_success_rate

pos_success, rot_success, combined = compute_success_rate(
    pos_errors,
    rot_errors,
    pos_threshold=0.005,  # 5mm
    rot_threshold=0.0873,  # 5 degrees
)

print(f"Position success: {pos_success * 100:.1f}%")
print(f"Rotation success: {rot_success * 100:.1f}%")
print(f"Combined success: {combined * 100:.1f}%")
```

#### 延迟测量

```python
from mfik.eval.metrics import measure_latency

def inference_fn():
    return solver.solve(target_pose, q_ref)

latency_stats = measure_latency(
    inference_fn,
    num_iterations=100,
    warmup_iterations=10,
    device="cuda",
)

print(f"Mean: {latency_stats['mean_ms']:.3f} ms")
print(f"95th percentile: {latency_stats['p95_ms']:.3f} ms")
```

#### 轨迹平滑度

```python
from mfik.eval.metrics import compute_trajectory_smoothness

smoothness = compute_trajectory_smoothness(
    trajectory=q_trajectory,
    dt=0.01,  # 时间步长
)

print(f"Max velocity: {smoothness['max_velocity']:.3f} rad/s")
print(f"Max acceleration: {smoothness['max_acceleration']:.3f} rad/s²")
print(f"Max jerk: {smoothness['max_jerk']:.3f} rad/s³")
```

#### 关节统计

```python
from mfik.eval.metrics import compute_joint_statistics

joint_stats = compute_joint_statistics(
    q_solutions,
    solver.joint_limits,
)

print(f"Violation rate: {joint_stats['total_violation_rate'] * 100:.2f}%")
print(f"Range utilization: {joint_stats['range_utilization']:.2f}")
```

#### 工作空间覆盖

```python
from mfik.eval.metrics import compute_workspace_coverage

# 获取末端位置
pred_poses = solver.fk.forward(q_solutions)
positions = pred_poses[:, :3]

coverage = compute_workspace_coverage(
    positions,
    grid_resolution=0.05,  # 5cm 网格
)

print(f"Workspace volume: {coverage['workspace_volume_m3']:.3f} m³")
print(f"Coverage ratio: {coverage['coverage_ratio']:.2f}")
```

### 3. 可视化工具 (`visualization.py`)

#### 误差分布图

```python
from mfik.eval.visualization import plot_error_distribution

fig = plot_error_distribution(
    pos_errors,
    rot_errors,
    pos_threshold=0.005,
    rot_threshold=0.0873,
    save_path="results/error_distribution.png",
)
```

生成包含直方图和箱线图的误差分布可视化。

#### 工作空间热力图

```python
from mfik.eval.visualization import plot_workspace_heatmap

fig = plot_workspace_heatmap(
    positions=positions,
    errors=pos_errors,
    plane="xy",  # 投影平面：'xy', 'xz', 'yz'
    grid_resolution=0.02,
    save_path="results/workspace_heatmap.png",
)
```

#### 轨迹跟踪可视化

```python
from mfik.eval.visualization import plot_trajectory_tracking

fig = plot_trajectory_tracking(
    target_trajectory=None,  # 可选的目标轨迹
    predicted_trajectory=q_trajectory,
    joint_names=["Joint 1", "Joint 2", ...],
    save_path="results/trajectory.png",
)
```

#### 方法对比

```python
from mfik.eval.visualization import plot_comparison

results_dict = {
    "MeanFlow": {"pos_errors": pos_errors_mf, "rot_errors": rot_errors_mf},
    "Jacobian": {"pos_errors": pos_errors_jac, "rot_errors": rot_errors_jac},
}

fig = plot_comparison(
    results_dict,
    metric_name="pos_error",
    save_path="results/comparison.png",
)
```

#### 延迟分布

```python
from mfik.eval.visualization import plot_latency_distribution
import numpy as np

latencies = np.array([...])  # 延迟数据（毫秒）
fig = plot_latency_distribution(
    latencies,
    save_path="results/latency.png",
)
```

#### 性能报告

```python
from mfik.eval.visualization import generate_performance_report

results = {
    "pos_error_mean": 0.002,
    "rot_error_mean": 0.05,
    "success_rate": (0.95, 0.93, 0.92),
    "avg_latency": 0.001,
    "throughput": 1000,
}

generate_performance_report(
    results,
    save_path="results/performance_report.md",
)
```

## 评估脚本

### 静态随机位姿测试

```bash
python scripts/eval/test_static_poses.py \
    --checkpoint checkpoints/panda_v1.pth \
    --urdf robots/panda_arm.urdf \
    --n-samples 10000 \
    --device cuda \
    --output-dir results/static_test
```

评估 10,000 个随机可达位姿的求解精度和速度。

### 连续轨迹测试

```bash
python scripts/eval/test_trajectories.py \
    --checkpoint checkpoints/panda_v1.pth \
    --urdf robots/panda_arm.urdf \
    --device cuda \
    --output-dir results/trajectory_test
```

测试直线、圆弧、螺旋线轨迹的跟踪性能。

### 数值 IK 对比

```bash
python scripts/eval/compare_numerical_ik.py \
    --checkpoint checkpoints/panda_v1.pth \
    --urdf robots/panda_arm.urdf \
    --n-samples 1000 \
    --device cuda \
    --output-dir results/comparison
```

与雅可比伪逆和阻尼最小二乘法对比。

## 完整评估流程示例

```python
from mfik.eval.inference import IKSolver
from mfik.eval.metrics import (
    compute_pose_error,
    compute_success_rate,
    measure_latency,
)
from mfik.eval.visualization import (
    plot_error_distribution,
    plot_workspace_heatmap,
    generate_performance_report,
)

# 1. 加载求解器
solver = IKSolver.from_checkpoint(
    "checkpoints/panda_v1.pth",
    "robots/panda_arm.urdf",
    device="cuda",
)

# 2. 生成测试数据
import torch
test_poses = torch.randn(10000, 7, device="cuda")
q_refs = torch.zeros(10000, 7, device="cuda")

# 3. 推理
q_solutions = solver.solve(test_poses, q_refs)

# 4. 计算指标
pos_errors, rot_errors = compute_pose_error(q_solutions, test_poses, solver.fk)
pos_success, rot_success, combined = compute_success_rate(pos_errors, rot_errors)

# 5. 基准测试
benchmark = solver.benchmark(test_poses[:100], q_refs[:100])

# 6. 可视化
plot_error_distribution(pos_errors, rot_errors, save_path="errors.png")

pred_poses = solver.fk.forward(q_solutions)
plot_workspace_heatmap(pred_poses[:, :3], pos_errors, save_path="heatmap.png")

# 7. 生成报告
results = {
    "pos_error_mean": pos_errors.mean().item(),
    "pos_error_95": torch.quantile(pos_errors, 0.95).item(),
    "success_rate": (pos_success, rot_success, combined),
    "avg_latency": benchmark["avg_latency"],
    "throughput": benchmark["throughput"],
}

generate_performance_report(results, "performance_report.md")
```

## 性能指标定义

| 指标 | 定义 | 目标值 |
|------|------|--------|
| Position Error | 末端位置的欧几里得距离（米） | < 5mm |
| Rotation Error | 四元数角度距离（弧度） | < 5° (0.0873 rad) |
| Success Rate | 同时满足位置和姿态阈值的比例 | > 95% |
| Latency | 单次推理耗时（毫秒） | < 1ms (GPU) |
| Throughput | 每秒推理次数 | > 1000 (GPU) |

## 与模型版本的集成

Eval 模块设计为版本无关：

```python
# 加载 v1 模型
solver_v1 = IKSolver.from_checkpoint("panda_v1.pth", urdf_path, device="cuda")

# 加载 v2 模型（未来）
solver_v2 = IKSolver.from_checkpoint("panda_v2.pth", urdf_path, device="cuda")

# 使用相同的评估工具
results_v1 = solver_v1.benchmark(test_poses)
results_v2 = solver_v2.benchmark(test_poses)

# 对比
plot_comparison(
    {"v1": results_v1, "v2": results_v2},
    metric_name="pos_error",
)
```

## 注意事项

1. **设备管理**：确保所有张量在同一设备上（CPU/GPU）
2. **四元数归一化**：目标位姿的四元数必须归一化
3. **关节限位**：默认启用关节限位裁剪，可通过 `clip_joints=False` 禁用
4. **轨迹跟踪**：使用 `solve_trajectory()` 时会自动管理参考关节状态
5. **内存管理**：批量推理时注意 GPU 内存占用

## API 参考

详细 API 文档请参考各模块的 docstring：

- `IKSolver`: 主推理接口
- `compute_pose_error()`: 位姿误差计算
- `compute_success_rate()`: 成功率统计
- `measure_latency()`: 延迟测量
- `plot_error_distribution()`: 误差可视化
- `plot_workspace_heatmap()`: 工作空间热力图
- `generate_performance_report()`: 性能报告生成
