"""
网格世界环境工具模块

该模块提供各种网格世界环境的工具函数，包括预设环境生成、值函数可视化等。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union, Any
import random
from matplotlib.animation import FuncAnimation

from .grid_environment import GridEnvironment, StochasticGridEnvironment


def create_default_grid() -> GridEnvironment:
    """
    创建默认的5x5网格世界环境
    
    返回:
    GridEnvironment: 默认网格环境
    """
    return GridEnvironment(
        height=5,
        width=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(1, 1), (2, 1), (3, 1)],
        traps=[(1, 3)],
        default_reward=-0.04,
        noise_prob=0.0
    )


def create_stochastic_grid() -> StochasticGridEnvironment:
    """
    创建默认的随机5x5网格世界环境
    
    返回:
    StochasticGridEnvironment: 默认随机网格环境
    """
    return StochasticGridEnvironment(
        height=5,
        width=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(1, 1), (2, 1), (3, 1)],
        traps=[(1, 3)],
        default_reward=-0.04,
        noise_prob=0.2
    )


def create_complex_grid(size: int = 10) -> GridEnvironment:
    """
    创建一个复杂的大型网格世界环境
    
    参数:
    size: 网格大小
    
    返回:
    GridEnvironment: 复杂网格环境
    """
    # 生成随机障碍物
    obstacles = []
    num_obstacles = size * 2
    
    for _ in range(num_obstacles):
        row = random.randint(1, size - 2)
        col = random.randint(1, size - 2)
        
        # 避免挡住起点和终点
        if (row, col) != (0, 0) and (row, col) != (size-1, size-1):
            obstacles.append((row, col))
    
    # 生成随机陷阱
    traps = []
    num_traps = size // 2
    
    for _ in range(num_traps):
        row = random.randint(1, size - 2)
        col = random.randint(1, size - 2)
        
        # 避免与障碍物重叠，以及挡住起点和终点
        if ((row, col) not in obstacles and 
            (row, col) != (0, 0) and 
            (row, col) != (size-1, size-1)):
            traps.append((row, col))
    
    return GridEnvironment(
        height=size,
        width=size,
        start_pos=(0, 0),
        goal_pos=(size-1, size-1),
        obstacles=obstacles,
        traps=traps,
        default_reward=-0.04,
        goal_reward=1.0,
        trap_reward=-1.0,
        noise_prob=0.0
    )


def create_maze_grid(height: int = 7, width: int = 7) -> GridEnvironment:
    """
    创建一个迷宫型网格世界环境
    
    参数:
    height: 网格高度
    width: 网格宽度
    
    返回:
    GridEnvironment: 迷宫网格环境
    """
    # 创建迷宫通道
    obstacles = []
    
    # 添加水平墙
    for row in range(1, height-1, 2):
        for col in range(width):
            if col != random.randint(0, width-1):  # 随机开一个口
                obstacles.append((row, col))
    
    # 添加垂直墙
    for col in range(1, width-1, 2):
        for row in range(height):
            if row != random.randint(0, height-1):  # 随机开一个口
                obstacles.append((row, col))
    
    # 确保起点和终点不是障碍物
    if (0, 0) in obstacles:
        obstacles.remove((0, 0))
    if (height-1, width-1) in obstacles:
        obstacles.remove((height-1, width-1))
    
    # 创建一些陷阱
    traps = []
    for _ in range(3):
        row = random.randint(1, height - 2)
        col = random.randint(1, width - 2)
        if (row, col) not in obstacles and (row, col) != (0, 0) and (row, col) != (height-1, width-1):
            traps.append((row, col))
    
    return GridEnvironment(
        height=height,
        width=width,
        start_pos=(0, 0),
        goal_pos=(height-1, width-1),
        obstacles=obstacles,
        traps=traps,
        default_reward=-0.04,
        goal_reward=1.0,
        trap_reward=-0.5,
        noise_prob=0.1
    )


def visualize_value_function(env: GridEnvironment, title: str = "Value Function") -> None:
    """
    可视化值函数
    
    参数:
    env: 网格环境
    title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制值函数热图
    plt.imshow(env.state_values, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Value')
    
    # 添加网格线
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.xticks(np.arange(-.5, env.width, 1), [])
    plt.yticks(np.arange(-.5, env.height, 1), [])
    
    # 在每个单元格显示值
    for row in range(env.height):
        for col in range(env.width):
            # 显示数值，保留两位小数
            plt.text(col, row, f"{env.state_values[row, col]:.2f}",
                     ha='center', va='center', color='black',
                     fontsize=8)
    
    # 标记特殊位置
    plt.plot(env.start_pos[1], env.start_pos[0], 'go', markersize=10, label='Start')
    plt.plot(env.goal_pos[1], env.goal_pos[0], 'b*', markersize=10, label='Goal')
    
    for trap in env.traps:
        plt.plot(trap[1], trap[0], 'rx', markersize=10)
    
    for obs in env.obstacles:
        plt.plot(obs[1], obs[0], 'ks', markersize=10)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_policy(env: GridEnvironment, title: str = "Policy") -> None:
    """
    可视化策略
    
    参数:
    env: 网格环境
    title: 图表标题
    """
    plt.figure(figsize=(8, 8))
    
    # 创建一个空白背景
    plt.imshow(np.zeros((env.height, env.width)), cmap='gray_r', alpha=0.1, interpolation='nearest')
    
    # 添加网格线
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.xticks(np.arange(-.5, env.width, 1), [])
    plt.yticks(np.arange(-.5, env.height, 1), [])
    
    # 创建方向符号映射
    directions = {
        env.UP: "↑",
        env.RIGHT: "→",
        env.DOWN: "↓",
        env.LEFT: "←"
    }
    
    # 绘制每个状态的策略
    for row in range(env.height):
        for col in range(env.width):
            if (row, col) in env.obstacles:
                plt.text(col, row, 'X', ha='center', va='center', fontsize=20, color='gray')
            elif (row, col) in env.traps:
                plt.text(col, row, 'T', ha='center', va='center', fontsize=20, color='red')
            elif (row, col) == env.goal_pos:
                plt.text(col, row, 'G', ha='center', va='center', fontsize=20, color='green')
            else:
                action = env.policy[row, col]
                plt.text(col, row, directions[action], ha='center', va='center', fontsize=20)
    
    # 标记起点
    plt.plot(env.start_pos[1], env.start_pos[0], 'go', markersize=10, label='Start')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def animate_trajectory(env: GridEnvironment, 
                     trajectory: List[Tuple[int, int]], 
                     interval: int = 500,
                     title: str = "Agent Trajectory") -> FuncAnimation:
    """
    创建智能体轨迹的动画
    
    参数:
    env: 网格环境
    trajectory: 位置坐标序列 [(row, col), ...]
    interval: 帧间隔（毫秒）
    title: 图表标题
    
    返回:
    FuncAnimation: 动画对象
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制环境
    ax.imshow(env.state_values, cmap='coolwarm', alpha=0.5, interpolation='nearest')
    
    # 添加网格线
    ax.grid(color='black', linestyle='-', linewidth=1)
    plt.xticks(np.arange(-.5, env.width, 1), [])
    plt.yticks(np.arange(-.5, env.height, 1), [])
    
    # 显示障碍物
    for obs in env.obstacles:
        row, col = obs
        ax.text(col, row, 'X', ha='center', va='center', fontsize=20, color='gray')
    
    # 显示陷阱
    for trap in env.traps:
        row, col = trap
        ax.text(col, row, 'T', ha='center', va='center', fontsize=20, color='red')
    
    # 显示目标
    row, col = env.goal_pos
    ax.text(col, row, 'G', ha='center', va='center', fontsize=20, color='green')
    
    # 初始化智能体
    agent = ax.plot([], [], 'bo', markersize=15)[0]
    
    # 轨迹线
    path, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
    
    def init():
        agent.set_data([], [])
        path.set_data([], [])
        return agent, path
    
    def update(frame):
        if frame < len(trajectory):
            row, col = trajectory[frame]
            agent.set_data([col], [row])
            
            # 更新轨迹线
            if frame > 0:
                traj_rows, traj_cols = zip(*trajectory[:frame+1])
                path.set_data(traj_cols, traj_rows)
        
        return agent, path
    
    ani = FuncAnimation(fig, update, frames=len(trajectory),
                      init_func=init, blit=True, interval=interval)
    
    plt.title(title)
    plt.tight_layout()
    
    return ani


def generate_optimal_trajectory(env: GridEnvironment) -> List[Tuple[int, int]]:
    """
    使用当前策略生成一条轨迹
    
    参数:
    env: 网格环境
    
    返回:
    List[Tuple[int, int]]: 轨迹坐标序列
    """
    # 保存当前环境状态
    original_pos = env.current_pos
    original_noise = env.noise_prob
    
    # 将环境设置为确定性
    env.noise_prob = 0.0
    
    # 重置环境
    env.reset()
    trajectory = [env.current_pos]
    done = False
    
    # 生成轨迹
    while not done and len(trajectory) < 100:  # 防止无限循环
        row, col = env.current_pos
        action = env.policy[row, col]
        _, _, done = env.step(action)
        trajectory.append(env.current_pos)
    
    # 恢复环境状态
    env.noise_prob = original_noise
    env.current_pos = original_pos
    
    return trajectory


def compare_value_functions(value_functions: List[np.ndarray], 
                          titles: List[str],
                          grid_shape: Tuple[int, int] = (5, 5)) -> None:
    """
    比较多个值函数
    
    参数:
    value_functions: 值函数列表
    titles: 标题列表
    grid_shape: 网格形状 (行, 列)
    """
    n = len(value_functions)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    if n == 1:
        axes = [axes]
    
    for i, (values, title) in enumerate(zip(value_functions, titles)):
        im = axes[i].imshow(values, cmap='coolwarm', interpolation='nearest')
        axes[i].set_title(title)
        
        # 添加网格线
        axes[i].grid(color='black', linestyle='-', linewidth=1)
        axes[i].set_xticks(np.arange(-.5, grid_shape[1], 1), [])
        axes[i].set_yticks(np.arange(-.5, grid_shape[0], 1), [])
        
        # 显示数值
        for row in range(grid_shape[0]):
            for col in range(grid_shape[1]):
                axes[i].text(col, row, f"{values[row, col]:.2f}",
                           ha='center', va='center', 
                           color='black', fontsize=8)
        
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 简单测试
    print("网格世界工具模块测试")
    
    # 创建默认环境
    env = create_default_grid()
    
    # 设置一个简单策略
    policy = np.zeros((env.height, env.width), dtype=int)
    for row in range(env.height):
        for col in range(env.width):
            if col < env.width - 1:
                policy[row, col] = env.RIGHT
            else:
                policy[row, col] = env.DOWN
    
    env.set_policy(policy)
    
    # 设置一些状态值
    values = np.zeros((env.height, env.width))
    for row in range(env.height):
        for col in range(env.width):
            distance = abs(row - env.goal_pos[0]) + abs(col - env.goal_pos[1])
            values[row, col] = 1.0 - 0.1 * distance
    
    env.set_state_values(values)
    
    # 可视化策略和值函数
    visualize_policy(env, "测试策略")
    visualize_value_function(env, "测试值函数")
    
    # 生成一条轨迹
    trajectory = generate_optimal_trajectory(env)
    print(f"生成轨迹: {trajectory}")
    
    # 创建轨迹动画
    ani = animate_trajectory(env, trajectory, interval=500, title="测试轨迹")
    plt.show()
    
    # 测试迷宫环境
    maze_env = create_maze_grid(7, 7)
    maze_env.set_policy(np.random.randint(0, 4, (7, 7)))
    
    # 设置一些随机状态值用于演示
    random_values = np.random.rand(7, 7)
    maze_env.set_state_values(random_values)
    
    visualize_policy(maze_env, "迷宫策略")
    visualize_value_function(maze_env, "迷宫值函数")
    
    # 比较不同的值函数
    value_functions = [
        values,
        np.random.rand(5, 5),
        np.zeros((5, 5))
    ]
    
    titles = ["距离启发式", "随机值", "零值"]
    
    compare_value_functions(value_functions, titles) 