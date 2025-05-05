"""
硬件加速贝尔曼求解器演示

该脚本演示了使用阻变存储器硬件加速的MBS算法和纯CPU版本在路径规划任务中的性能比较。
根据论文"Memristive Bellman solver for decision-making"实现，展示了精确解和近似解之间的差异。
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入MBSolver和网格环境
from mbs_core.bellman_solver import MBSolver
from grid_world.grid_environment import GridEnvironment
from grid_world.utils import visualize_value_function, visualize_policy, animate_trajectory


def create_maze_world(size: int = 5, random_obstacles: bool = False) -> GridEnvironment:
    """
    创建5×5迷宫世界环境(对应论文Figure 4a)
    
    参数:
    size: 网格大小
    random_obstacles: 是否生成随机障碍物
    
    返回:
    GridEnvironment: 网格环境
    """
    # 设置起点、终点和奖励点
    start_pos = (0, 0)           # state_1 (论文中的Start)
    goal_pos = (size-1, size-1)  # state_25 (论文中的End)
    bonus_pos = (2, 1)           # state_12 (论文中的Bonus)
    
    # 设置陷阱位置(论文中的Trap)
    trap_pos = [(2, 2), (3, 0)]  # state_13和state_16
    
    # 设置障碍物位置
    if random_obstacles:
        obstacles = []
        num_obstacles = size // 2
        
        for _ in range(num_obstacles):
            row = np.random.randint(1, size-1)
            col = np.random.randint(1, size-1)
            
            # 确保不会阻塞起点和终点
            if (row, col) != start_pos and (row, col) != goal_pos and (row, col) != bonus_pos:
                obstacles.append((row, col))
    else:
        # 使用论文中的固定障碍物配置
        obstacles = trap_pos
    
    # 创建环境
    env = GridEnvironment(
        height=size,
        width=size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        default_reward=-0.04,  # 默认奖励
        goal_reward=1.0,       # 终点奖励
        noise_prob=0.0         # 确定性环境
    )
    
    # 设置特殊奖励点(Bonus)
    env.set_cell_reward(bonus_pos, 0.5)
    
    # 设置陷阱(负奖励)
    for trap in trap_pos:
        env.set_cell_reward(trap, -0.5)
    
    return env


def create_road_map(width: int = 6, height: int = 4) -> GridEnvironment:
    """
    创建道路地图环境(对应论文Figure 4g)
    
    参数:
    width: 地图宽度
    height: 地图高度
    
    返回:
    GridEnvironment: 网格环境
    """
    # 设置起点和终点
    start_pos = (0, 0)         # state_1 (论文中的Start)
    goal_pos = (height-1, width-2)  # state_18 (论文中的End)
    
    # 设置湖泊障碍物(对应论文中的Lake)
    obstacles = [
        (1, 1), (1, 2), (1, 3),  # 中心湖泊
        (2, 1), (2, 2), (2, 3),
    ]
    
    # 创建环境
    env = GridEnvironment(
        height=height,
        width=width,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        default_reward=-0.04,  # 默认奖励
        goal_reward=1.0,       # 终点奖励
        noise_prob=0.0         # 确定性环境
    )
    
    return env


def grid_to_matrices(env: GridEnvironment) -> Tuple[np.ndarray, np.ndarray]:
    """
    将网格环境转换为MBS所需的状态转移和奖励矩阵
    
    参数:
    env: 网格环境
    
    返回:
    Tuple[np.ndarray, np.ndarray]: (奖励向量, 状态转移概率矩阵)
    """
    # 获取状态转移矩阵
    P = env.get_transition_matrix()
    
    # 获取奖励向量
    R = env.get_reward_vector()
    
    return R, P


def solve_grid_world(env: GridEnvironment, 
                    use_hardware: bool = False, 
                    enable_noise: bool = False,
                    noise_level: float = 0.01,  # 噪声级别（论文中的φ值）
                    config_path: str = None,
                    max_iterations: int = 20) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    使用MBSolver求解网格世界
    
    参数:
    env: 网格环境
    use_hardware: 是否使用硬件加速
    enable_noise: 是否启用噪声（用于近似解）
    noise_level: 噪声级别（论文中的φ值，标准差）
    config_path: 配置文件路径
    max_iterations: 最大迭代次数
    
    返回:
    Tuple[np.ndarray, np.ndarray, Dict[str, any]]: (值向量, 权重矩阵, 求解统计)
    """
    # 获取奖励向量和状态转移矩阵
    R, P = grid_to_matrices(env)
    
    # 创建求解器
    state_size = env.height * env.width
    solver = MBSolver(
        state_size=state_size, 
        gamma=0.9,            # 折扣因子（论文中的γ）
        epsilon=0.1,          # MBr收敛阈值（论文中的ε）
        tau=0.1,              # 权重矩阵收敛阈值（论文中的τ）
        use_hardware=use_hardware,
        noise_level=noise_level,  # 噪声级别（论文中的φ值）
        enable_noise=enable_noise, # 是否启用噪声
        config_path=config_path
    )
    
    # 预处理
    solver.preprocess(R, P)
    
    # 计时求解过程
    start_time = time.time()
    V, W, stats = solver.solve(max_iterations=max_iterations)
    end_time = time.time()
    
    # 计算求解时间
    solution_time = end_time - start_time
    
    # 补充统计信息
    stats["solution_time"] = solution_time
    stats["env_size"] = state_size
    stats["noise_level"] = noise_level
    stats["noise_enabled"] = enable_noise
    stats["hardware_used"] = use_hardware
    
    # 清理资源
    solver.cleanup()
    
    return V, W, stats


def update_env_with_results(env: GridEnvironment, V: np.ndarray, W: np.ndarray) -> None:
    """
    用求解结果更新环境的值函数和策略
    
    参数:
    env: 网格环境
    V: 值向量
    W: 权重矩阵
    """
    # 将V重塑为网格形状
    V_grid = V.reshape(env.height, env.width)
    
    # 设置环境的状态值
    env.set_state_values(V_grid)
    
    # 从权重矩阵中提取策略
    policy = np.zeros((env.height, env.width), dtype=int)
    
    for s in range(V.shape[0]):
        row, col = s // env.width, s % env.width
        
        if (row, col) == env.goal_pos or (row, col) in env.obstacles:
            continue
        
        # 找出最可能转移到的下一个状态
        next_states = np.where(W[s] > 0)[0]
        if len(next_states) > 0:
            best_next = next_states[np.argmax(W[s, next_states])]
            next_row, next_col = best_next // env.width, best_next % env.width
            
            # 确定动作
            if next_row < row:
                action = env.UP
            elif next_row > row:
                action = env.DOWN
            elif next_col < col:
                action = env.LEFT
            elif next_col > col:
                action = env.RIGHT
            else:
                action = 0  # 默认
            
            policy[row, col] = action
    
    # 设置环境的策略
    env.set_policy(policy)


def visualize_transition_matrix(W: np.ndarray, title: str = "状态转移概率矩阵") -> None:
    """
    可视化状态转移矩阵（权重矩阵）
    
    参数:
    W: 权重矩阵
    title: 图表标题
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(W, cmap='viridis')
    plt.colorbar(label='转移概率')
    plt.title(title)
    plt.xlabel('目标状态')
    plt.ylabel('源状态')
    plt.tight_layout()
    plt.show()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MBS硬件加速演示程序")
    parser.add_argument("--task", type=str, default="maze", choices=["maze", "road"],
                       help="任务类型：maze(5×5迷宫)或road(道路地图)")
    parser.add_argument("--size", type=int, default=5,
                       help="迷宫大小(仅maze任务)")
    parser.add_argument("--hardware", action="store_true",
                       help="使用硬件加速")
    parser.add_argument("--compare", action="store_true",
                       help="比较精确解和近似解")
    parser.add_argument("--visualize", action="store_true",
                       help="可视化结果")
    parser.add_argument("--noise_level", type=float, default=0.01,
                       help="噪声级别(φ值，控制标准差)")
    parser.add_argument("--noise_test", action="store_true",
                       help="执行噪声影响测试")
    parser.add_argument("--output_file", type=str, default=None,
                       help="结果输出文件")
    
    args = parser.parse_args()
    
    # 打印标题
    print("\n" + "="*50)
    print("阻变存储器贝尔曼求解器演示程序")
    print("基于论文 'Memristive Bellman solver for decision-making'")
    print("="*50 + "\n")
    
    # 创建环境
    if args.task == "maze":
        print(f"创建 {args.size}×{args.size} 迷宫环境...")
        env = create_maze_world(size=args.size)
        task_name = f"{args.size}×{args.size} 迷宫"
    else:
        print("创建道路地图环境...")
        env = create_road_map()
        task_name = "道路地图"
    
    # 准备结果存储
    results = []
    
    if args.noise_test:
        # 执行噪声影响测试
        print("\n执行噪声影响测试...")
        noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        
        # 记录每个噪声级别的性能
        noise_test_results = []
        
        for noise in noise_levels:
            # 仅在噪声为0时禁用噪声，其他情况启用
            enable_noise = noise > 0
            
            print(f"\n测试噪声级别 φ={noise:.3f} {'(禁用噪声)' if not enable_noise else ''}")
            
            # 求解
            V, W, stats = solve_grid_world(
                env=env, 
                use_hardware=args.hardware,
                enable_noise=enable_noise, 
                noise_level=noise
            )
            
            # 记录结果
            noise_test_results.append({
                "noise_level": noise,
                "mbr_iterations": stats["mbr_iterations"],
                "mbdot_operations": stats["mbdot_operations"],
                "solution_time": stats["solution_time"],
                "policy_iterations": stats["policy_iterations"]
            })
            
            # 打印当前结果
            print(f"  噪声级别(φ): {noise:.3f}")
            print(f"  MBr迭代次数: {stats['mbr_iterations']}")
            print(f"  MBdot操作次数: {stats['mbdot_operations']}")
            print(f"  策略迭代次数: {stats['policy_iterations']}")
            print(f"  求解时间: {stats['solution_time']:.3f}秒")
        
        # 转换为DataFrame并保存
        import pandas as pd
        df = pd.DataFrame(noise_test_results)
        
        # 如果指定了输出文件，保存结果
        if args.output_file:
            output_file = args.output_file
        else:
            timestamp = int(time.time())
            env_size = f"{args.size}x{args.size}" if args.task == "maze" else "road"
            output_file = f"{env_size}_noise_test_{timestamp}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"\n测试结果已保存到 {output_file}")
        
        # 如果启用可视化，绘制噪声影响曲线
        if args.visualize:
            plt.figure(figsize=(12, 6))
            
            # 绘制MBr迭代次数
            plt.subplot(1, 2, 1)
            plt.plot(df["noise_level"], df["mbr_iterations"], 'o-', linewidth=2)
            plt.xlabel("噪声级别(φ)")
            plt.ylabel("MBr迭代次数")
            plt.title("噪声对MBr迭代次数的影响")
            plt.grid(True)
            
            # 绘制求解时间
            plt.subplot(1, 2, 2)
            plt.plot(df["noise_level"], df["solution_time"], 'o-', linewidth=2)
            plt.xlabel("噪声级别(φ)")
            plt.ylabel("求解时间(秒)")
            plt.title("噪声对求解时间的影响")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{env_size}_noise_impact_{timestamp}.png")
            plt.show()
            
    elif args.compare:
        # 比较精确解和近似解
        print("\n计算精确解(无噪声)...")
        V_exact, W_exact, stats_exact = solve_grid_world(
            env=env, 
            use_hardware=args.hardware,
            enable_noise=False
        )
        
        print("\n计算近似解(有噪声)...")
        V_approx, W_approx, stats_approx = solve_grid_world(
            env=env, 
            use_hardware=args.hardware,
            enable_noise=True,
            noise_level=args.noise_level
        )
        
        # 比较两种解法
        print("\n比较精确解和近似解:")
        print(f"  精确解MBr迭代次数: {stats_exact['mbr_iterations']}")
        print(f"  近似解MBr迭代次数: {stats_approx['mbr_iterations']}")
        print(f"  迭代次数减少: {stats_exact['mbr_iterations'] - stats_approx['mbr_iterations']} ({100*(stats_exact['mbr_iterations']-stats_approx['mbr_iterations'])/stats_exact['mbr_iterations']:.1f}%)")
        print(f"  精确解求解时间: {stats_exact['solution_time']:.3f}秒")
        print(f"  近似解求解时间: {stats_approx['solution_time']:.3f}秒")
        print(f"  时间减少: {stats_exact['solution_time'] - stats_approx['solution_time']:.3f}秒 ({100*(stats_exact['solution_time']-stats_approx['solution_time'])/stats_exact['solution_time']:.1f}%)")
        
        # 计算值函数差异
        value_diff = np.mean((V_exact - V_approx) ** 2)
        print(f"  值函数均方差: {value_diff:.6f}")
        
        # 计算策略差异
        policy_exact = np.argmax(W_exact, axis=1)
        policy_approx = np.argmax(W_approx, axis=1)
        policy_diff = np.sum(policy_exact != policy_approx) / len(policy_exact)
        print(f"  策略差异率: {policy_diff:.2%}")
        
        # 将结果更新到环境
        env.set_value_function(V_exact)
        
        # 计算精确解路径
        path_exact = env.compute_optimal_path()
        print("\n精确解最优路径:")
        print(f"  长度: {len(path_exact)-1} 步")
        
        # 更新为近似解
        env.set_value_function(V_approx)
        path_approx = env.compute_optimal_path()
        print("\n近似解最优路径:")
        print(f"  长度: {len(path_approx)-1} 步")
        
        # 记录详细比较结果
        compare_results = {
            "task": task_name,
            "env_size": stats_exact["env_size"],
            "noise_level": args.noise_level,
            "hardware_used": args.hardware,
            "exact_mbr_iterations": stats_exact["mbr_iterations"],
            "approx_mbr_iterations": stats_approx["mbr_iterations"],
            "exact_solution_time": stats_exact["solution_time"],
            "approx_solution_time": stats_approx["solution_time"],
            "iteration_reduction_pct": 100*(stats_exact["mbr_iterations"]-stats_approx["mbr_iterations"])/stats_exact["mbr_iterations"],
            "time_reduction_pct": 100*(stats_exact["solution_time"]-stats_approx["solution_time"])/stats_exact["solution_time"],
            "value_function_mse": value_diff,
            "policy_difference_rate": policy_diff,
            "exact_path_length": len(path_exact)-1,
            "approx_path_length": len(path_approx)-1
        }
        
        # 如果指定了输出文件，保存结果
        if args.output_file:
            # 如果文件已存在，追加，否则创建新文件
            import os
            import pandas as pd
            
            df = pd.DataFrame([compare_results])
            
            if os.path.exists(args.output_file):
                df.to_csv(args.output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(args.output_file, index=False)
                
            print(f"\n比较结果已保存到 {args.output_file}")
        
        # 可视化结果
        if args.visualize:
            # 可视化值函数
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            visualize_value_function(env, V_exact, title="精确解值函数")
            
            plt.subplot(1, 2, 2)
            visualize_value_function(env, V_approx, title="近似解值函数")
            
            plt.tight_layout()
            plt.savefig(f"value_functions_{int(time.time())}.png")
            plt.show()
            
            # 可视化策略
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            env.set_value_function(V_exact)
            visualize_policy(env, title="精确解策略")
            
            plt.subplot(1, 2, 2)
            env.set_value_function(V_approx)
            visualize_policy(env, title="近似解策略")
            
            plt.tight_layout()
            plt.savefig(f"policies_{int(time.time())}.png")
            plt.show()
            
            # 可视化路径
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            animate_trajectory(env, path_exact, title=f"精确解路径({len(path_exact)-1}步)")
            
            plt.subplot(1, 2, 2)
            animate_trajectory(env, path_approx, title=f"近似解路径({len(path_approx)-1}步)")
            
            plt.tight_layout()
            plt.savefig(f"paths_{int(time.time())}.png")
            plt.show()
    
    else:
        # 仅求解一次
        print(f"\n使用{'近似解(有噪声)' if args.noise_level > 0 else '精确解(无噪声)'}求解...")
        
        # 设置噪声参数
        enable_noise = args.noise_level > 0
        
        # 求解
        V, W, stats = solve_grid_world(
            env=env, 
            use_hardware=args.hardware,
            enable_noise=enable_noise,
            noise_level=args.noise_level
        )
        
        # 打印结果
        print("\n求解结果:")
        print(f"  MBr迭代次数: {stats['mbr_iterations']}")
        print(f"  MBdot操作次数: {stats['mbdot_operations']}")
        print(f"  策略迭代次数: {stats['policy_iterations']}")
        print(f"  求解时间: {stats['solution_time']:.3f}秒")
        
        # 更新环境的值函数
        env.set_value_function(V)
        
        # 计算最优路径
        path = env.compute_optimal_path()
        print(f"\n最优路径长度: {len(path)-1} 步")
        
        # 可视化结果
        if args.visualize:
            # 可视化值函数
            plt.figure(figsize=(6, 5))
            visualize_value_function(env, V, title="值函数")
            plt.savefig(f"value_function_{int(time.time())}.png")
            plt.show()
            
            # 可视化策略
            plt.figure(figsize=(6, 5))
            visualize_policy(env, title="最优策略")
            plt.savefig(f"policy_{int(time.time())}.png")
            plt.show()
            
            # 可视化路径
            plt.figure(figsize=(6, 5))
            animate_trajectory(env, path, title=f"最优路径({len(path)-1}步)")
            plt.savefig(f"path_{int(time.time())}.png")
            plt.show()
    
    print("\n演示完成!")


# 调用主函数
if __name__ == "__main__":
    main() 