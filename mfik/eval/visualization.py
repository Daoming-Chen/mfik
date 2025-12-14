"""
Visualization Tools

Provides plotting and visualization functions for IK evaluation:
- Accuracy distribution plots (boxplots, histograms)
- Workspace heatmaps
- Trajectory tracking visualization
"""

from typing import Optional, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_error_distribution(
    pos_errors: torch.Tensor,
    rot_errors: torch.Tensor,
    pos_threshold: float = 0.005,
    rot_threshold: float = 0.0873,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot position and rotation error distributions.
    
    Args:
        pos_errors: Position errors [N] in meters
        rot_errors: Rotation errors [N] in radians
        pos_threshold: Position success threshold (for reference line)
        rot_threshold: Rotation success threshold (for reference line)
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    pos_errors = pos_errors.cpu().numpy() * 1000  # Convert to mm
    rot_errors = rot_errors.cpu().numpy() * np.pi / 180  # Convert to degrees
    pos_threshold_mm = pos_threshold * 1000
    rot_threshold_deg = rot_threshold * 180 / np.pi
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position error histogram
    ax = axes[0, 0]
    ax.hist(pos_errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(pos_threshold_mm, color='r', linestyle='--', 
               label=f'Threshold ({pos_threshold_mm:.1f} mm)')
    ax.set_xlabel('Position Error (mm)')
    ax.set_ylabel('Count')
    ax.set_title('Position Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotation error histogram
    ax = axes[0, 1]
    ax.hist(rot_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(rot_threshold_deg, color='r', linestyle='--',
               label=f'Threshold ({rot_threshold_deg:.1f}°)')
    ax.set_xlabel('Rotation Error (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Rotation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position error boxplot
    ax = axes[1, 0]
    box = ax.boxplot([pos_errors], vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')
    ax.axhline(pos_threshold_mm, color='r', linestyle='--', 
               label=f'Threshold')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Position Error Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotation error boxplot
    ax = axes[1, 1]
    box = ax.boxplot([rot_errors], vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('lightyellow')
    ax.axhline(rot_threshold_deg, color='r', linestyle='--',
               label=f'Threshold')
    ax.set_ylabel('Rotation Error (degrees)')
    ax.set_title('Rotation Error Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics
    pos_stats = f"Mean: {pos_errors.mean():.2f} mm, Median: {np.median(pos_errors):.2f} mm, 95%: {np.percentile(pos_errors, 95):.2f} mm"
    rot_stats = f"Mean: {rot_errors.mean():.2f}°, Median: {np.median(rot_errors):.2f}°, 95%: {np.percentile(rot_errors, 95):.2f}°"
    
    fig.suptitle(f'IK Solution Accuracy\n{pos_stats}\n{rot_stats}', 
                 fontsize=10, y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error distribution plot to {save_path}")
    
    return fig


def plot_workspace_heatmap(
    positions: torch.Tensor,
    errors: torch.Tensor,
    plane: str = "xy",
    grid_resolution: float = 0.02,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot workspace heatmap showing error distribution in 2D plane.
    
    Args:
        positions: End-effector positions [N, 3]
        errors: Position or rotation errors [N]
        plane: Projection plane ('xy', 'xz', or 'yz')
        grid_resolution: Grid cell size in meters
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    positions = positions.cpu().numpy()
    errors = errors.cpu().numpy()
    
    # Select plane axes
    plane_map = {
        "xy": (0, 1, 2, "X (m)", "Y (m)"),
        "xz": (0, 2, 1, "X (m)", "Z (m)"),
        "yz": (1, 2, 0, "Y (m)", "Z (m)"),
    }
    
    if plane not in plane_map:
        raise ValueError(f"Invalid plane '{plane}'. Choose from {list(plane_map.keys())}")
    
    i, j, k, xlabel, ylabel = plane_map[plane]
    
    # Project to 2D
    x = positions[:, i]
    y = positions[:, j]
    
    # Create grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_bins = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    y_bins = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    
    # Compute average error in each cell
    grid = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)
    
    for idx in range(len(x)):
        xi = np.searchsorted(x_bins, x[idx]) - 1
        yi = np.searchsorted(y_bins, y[idx]) - 1
        
        if 0 <= xi < len(x_bins) - 1 and 0 <= yi < len(y_bins) - 1:
            if np.isnan(grid[yi, xi]):
                grid[yi, xi] = errors[idx]
            else:
                grid[yi, xi] = (grid[yi, xi] + errors[idx]) / 2
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        grid,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        cmap='viridis',
        aspect='auto',
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Position Error (m)')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Workspace Error Heatmap ({plane.upper()} plane)')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved workspace heatmap to {save_path}")
    
    return fig


def plot_trajectory_tracking(
    target_trajectory: torch.Tensor,
    predicted_trajectory: torch.Tensor,
    joint_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot joint trajectory tracking results.
    
    Args:
        target_trajectory: Target joint angles [T, n_joints] (if available)
        predicted_trajectory: Predicted joint angles [T, n_joints]
        joint_names: Optional joint names
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    target_trajectory = target_trajectory.cpu().numpy() if target_trajectory is not None else None
    predicted_trajectory = predicted_trajectory.cpu().numpy()
    
    T, n_joints = predicted_trajectory.shape
    time = np.arange(T)
    
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(n_joints)]
    
    # Create subplots
    n_cols = 3
    n_rows = (n_joints + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i in range(n_joints):
        ax = axes[i]
        
        if target_trajectory is not None:
            ax.plot(time, target_trajectory[:, i], 'b--', label='Target', alpha=0.7)
        ax.plot(time, predicted_trajectory[:, i], 'r-', label='Predicted', linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Angle (rad)')
        ax.set_title(joint_names[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_joints, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle('Joint Trajectory Tracking', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    
    return fig


def plot_comparison(
    results_dict: dict,
    metric_name: str = "pos_error",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot comparison between multiple methods.
    
    Args:
        results_dict: Dict mapping method names to result dictionaries
                      Each result dict should have 'pos_errors' and 'rot_errors'
        metric_name: Metric to compare ('pos_error' or 'rot_error')
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    ax = axes[0]
    data_to_plot = []
    labels = []
    
    for method_name, results in results_dict.items():
        if metric_name == "pos_error":
            errors = results["pos_errors"].cpu().numpy() * 1000  # to mm
            unit = "mm"
        else:
            errors = results["rot_errors"].cpu().numpy() * 180 / np.pi  # to degrees
            unit = "degrees"
        
        data_to_plot.append(errors)
        labels.append(method_name)
    
    box = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel(f'{"Position" if metric_name == "pos_error" else "Rotation"} Error ({unit})')
    ax.set_title('Method Comparison (Boxplot)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bar chart of statistics
    ax = axes[1]
    methods = list(results_dict.keys())
    means = []
    medians = []
    p95s = []
    
    for method_name, results in results_dict.items():
        if metric_name == "pos_error":
            errors = results["pos_errors"].cpu().numpy() * 1000
        else:
            errors = results["rot_errors"].cpu().numpy() * 180 / np.pi
        
        means.append(np.mean(errors))
        medians.append(np.median(errors))
        p95s.append(np.percentile(errors, 95))
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, means, width, label='Mean', alpha=0.8)
    ax.bar(x, medians, width, label='Median', alpha=0.8)
    ax.bar(x + width, p95s, width, label='95th %ile', alpha=0.8)
    
    ax.set_ylabel(f'Error ({unit})')
    ax.set_title('Method Comparison (Statistics)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    return fig


def plot_latency_distribution(
    latencies: np.ndarray,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot inference latency distribution.
    
    Args:
        latencies: Array of latencies in milliseconds [N]
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(latencies), color='r', linestyle='--', 
               label=f'Mean: {np.mean(latencies):.2f} ms')
    ax.axvline(np.median(latencies), color='g', linestyle='--',
               label=f'Median: {np.median(latencies):.2f} ms')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Percentile plot
    ax = axes[1]
    percentiles = np.arange(0, 101, 1)
    values = np.percentile(latencies, percentiles)
    ax.plot(percentiles, values, linewidth=2)
    ax.axhline(np.percentile(latencies, 95), color='r', linestyle='--',
               label=f'95th: {np.percentile(latencies, 95):.2f} ms')
    ax.axhline(np.percentile(latencies, 99), color='orange', linestyle='--',
               label=f'99th: {np.percentile(latencies, 99):.2f} ms')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Percentiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved latency plot to {save_path}")
    
    return fig


def generate_performance_report(
    results: dict,
    save_path: str,
):
    """
    Generate a comprehensive performance report in Markdown format.
    
    Args:
        results: Dictionary containing evaluation results
        save_path: Path to save the report (.md file)
    """
    report = []
    report.append("# IK Solver Performance Report\n")
    
    # Accuracy metrics
    if "pos_error_mean" in results:
        report.append("## Accuracy Metrics\n")
        report.append("| Metric | Position Error (mm) | Rotation Error (°) |")
        report.append("|--------|---------------------|-------------------|")
        report.append(
            f"| Mean | {results.get('pos_error_mean', 0) * 1000:.2f} | "
            f"{results.get('rot_error_mean', 0) * 180 / np.pi:.2f} |"
        )
        report.append(
            f"| Median | {results.get('pos_error_median', 0) * 1000:.2f} | "
            f"{results.get('rot_error_median', 0) * 180 / np.pi:.2f} |"
        )
        report.append(
            f"| 95th %ile | {results.get('pos_error_95', 0) * 1000:.2f} | "
            f"{results.get('rot_error_95', 0) * 180 / np.pi:.2f} |"
        )
        report.append("")
    
    # Success rate
    if "success_rate" in results:
        report.append("## Success Rate\n")
        report.append(f"- Position: {results['success_rate'][0] * 100:.1f}%")
        report.append(f"- Rotation: {results['success_rate'][1] * 100:.1f}%")
        report.append(f"- Combined: {results['success_rate'][2] * 100:.1f}%")
        report.append("")
    
    # Latency metrics
    if "avg_latency" in results:
        report.append("## Latency Metrics\n")
        report.append(f"- Average: {results['avg_latency'] * 1000:.2f} ms")
        report.append(f"- Throughput: {results.get('throughput', 0):.1f} inferences/sec")
        report.append("")
    
    # Write to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Performance report saved to {save_path}")
