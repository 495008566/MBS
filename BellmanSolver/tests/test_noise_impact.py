"""
噪声影响测试

此测试模块验证阻变存储器内在噪声对贝尔曼方程求解性能的影响，
根据论文"Memristive Bellman solver for decision-making"中的核心创新点，
噪声水平φ的不同取值会影响收敛速度和解的质量。
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入MBSolver和测试环境
from mbs_core.bellman_solver import MBSolver
from grid_world.grid_environment import GridEnvironment


def create_test_environment(env_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建测试环境的奖励向量和状态转移概率矩阵
    
    参数:
    env_size: 环境大小（方形网格边长）
    
    返回:
    Tuple[np.ndarray, np.ndarray]: (奖励向量, 状态转移概率矩阵)
    """
    state_size = env_size * env_size
    
    # 创建奖励向量，目标在右下角
    R = np.zeros(state_size)
    R[state_size-1] = 1.0  # 右下角作为目标
    
    # 创建状态转移矩阵
    P = np.zeros((state_size, state_size))
    
    # 添加四个方向的转移（上下左右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for s in range(state_size):
        row, col = s // env_size, s % env_size
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            # 检查是否在网格内
            if 0 <= new_row < env_size and 0 <= new_col < env_size:
                new_s = new_row * env_size + new_col
                P[s, new_s] = 0.25  # 等概率
    
    return R, P


def test_noise_impact(env_size: int = 5, 
                      noise_levels: List[float] = None, 
                      trials: int = 3,
                      output_file: str = None,
                      visualize: bool = True) -> pd.DataFrame:
    """
    测试不同噪声水平对求解性能的影响
    
    参数:
    env_size: 环境大小
    noise_levels: 要测试的噪声水平列表
    trials: 每个噪声水平的重复测试次数
    output_file: 输出文件名
    visualize: 是否可视化结果
    
    返回:
    pd.DataFrame: 测试结果
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    
    state_size = env_size * env_size
    R, P = create_test_environment(env_size)
    
    # 先计算一个精确解作为基准
    print("计算精确解作为基准...")
    solver_exact = MBSolver(
        state_size=state_size,
        gamma=0.9,
        epsilon=0.1,
        tau=0.1,
        enable_noise=False
    )
    solver_exact.preprocess(R, P)
    V_exact, W_exact, stats_exact = solver_exact.solve(max_iterations=200)
    
    # 记录结果
    results = []
    
    for noise_level in noise_levels:
        print(f"\n测试噪声水平 φ={noise_level:.4f}")
        
        for trial in range(trials):
            # 是否启用噪声
            enable_noise = noise_level > 0
            
            # 创建求解器
            solver = MBSolver(
                state_size=state_size,
                gamma=0.9,
                epsilon=0.1,
                tau=0.1,
                enable_noise=enable_noise,
                noise_level=noise_level
            )
            
            # 预处理
            solver.preprocess(R, P)
            
            # 求解
            start_time = time.time()
            V, W, stats = solver.solve(max_iterations=200)
            solution_time = time.time() - start_time
            
            # 计算与精确解的差异
            value_mse = np.mean((V - V_exact) ** 2)
            
            # 计算策略差异率
            policy = np.argmax(W, axis=1)
            policy_exact = np.argmax(W_exact, axis=1)
            policy_diff = np.sum(policy != policy_exact) / state_size
            
            # 记录结果
            results.append({
                "noise_level": noise_level,
                "trial": trial + 1,
                "mbr_iterations": stats["mbr_iterations"],
                "mbdot_operations": stats["mbdot_operations"],
                "solution_time": solution_time,
                "value_mse": value_mse,
                "policy_diff_rate": policy_diff
            })
            
            print(f"  试验 {trial+1}: MBr迭代={stats['mbr_iterations']}, " +
                  f"时间={solution_time:.3f}秒, 策略差异率={policy_diff:.2%}")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 按噪声水平分组计算平均值
    df_avg = df.groupby("noise_level").mean().reset_index()
    
    # 保存结果
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到 {output_file}")
        
        # 保存平均结果
        avg_file = output_file.replace(".csv", "_avg.csv")
        df_avg.to_csv(avg_file, index=False)
        print(f"平均结果已保存到 {avg_file}")
    
    # 可视化结果
    if visualize:
        plot_noise_impact(df_avg)
    
    return df


def plot_noise_impact(df: pd.DataFrame) -> None:
    """
    可视化噪声影响
    
    参数:
    df: 包含测试结果的DataFrame
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制MBr迭代次数
    plt.subplot(2, 2, 1)
    plt.plot(df["noise_level"], df["mbr_iterations"], 'o-', linewidth=2)
    plt.xlabel("噪声水平(φ)")
    plt.ylabel("MBr迭代次数")
    plt.title("噪声对MBr迭代次数的影响")
    plt.grid(True)
    
    # 绘制求解时间
    plt.subplot(2, 2, 2)
    plt.plot(df["noise_level"], df["solution_time"], 'o-', linewidth=2)
    plt.xlabel("噪声水平(φ)")
    plt.ylabel("求解时间(秒)")
    plt.title("噪声对求解时间的影响")
    plt.grid(True)
    
    # 绘制值函数MSE
    plt.subplot(2, 2, 3)
    plt.plot(df["noise_level"], df["value_mse"], 'o-', linewidth=2)
    plt.xlabel("噪声水平(φ)")
    plt.ylabel("值函数均方差")
    plt.title("噪声对值函数精度的影响")
    plt.grid(True)
    
    # 绘制策略差异率
    plt.subplot(2, 2, 4)
    plt.plot(df["noise_level"], df["policy_diff_rate"], 'o-', linewidth=2)
    plt.xlabel("噪声水平(φ)")
    plt.ylabel("策略差异率")
    plt.title("噪声对决策质量的影响")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"noise_impact_{int(time.time())}.png")
    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="噪声影响测试")
    parser.add_argument("--size", type=int, default=5, help="环境大小")
    parser.add_argument("--trials", type=int, default=3, help="每个噪声水平的测试次数")
    parser.add_argument("--output", type=str, default=None, help="输出文件名")
    parser.add_argument("--no-visual", action="store_true", help="禁用可视化")
    
    args = parser.parse_args()
    
    print(f"开始测试噪声对{args.size}×{args.size}环境的影响...")
    print(f"每个噪声水平重复{args.trials}次...")
    
    output_file = args.output
    if output_file is None:
        output_file = f"{args.size}x{args.size}_noise_experiment_results{int(time.time())}.csv"
    
    test_noise_impact(
        env_size=args.size,
        trials=args.trials,
        output_file=output_file,
        visualize=not args.no_visual
    )
    
    print("\n测试完成!")


if __name__ == "__main__":
    main() 