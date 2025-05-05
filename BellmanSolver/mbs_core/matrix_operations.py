"""
矩阵操作模块

该模块提供矩阵处理的各种工具函数，包括矩阵规范化、稀疏矩阵转换、
特征值计算等，用于支持贝尔曼解算器的核心算法。
"""

import numpy as np
from typing import Tuple, Optional, List, Union


def normalize_matrix(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    归一化矩阵，使每行或每列的和为1
    
    参数:
    matrix: 输入矩阵
    axis: 归一化轴，1表示行，0表示列
    
    返回:
    np.ndarray: 归一化后的矩阵
    """
    sums = matrix.sum(axis=axis, keepdims=True)
    # 避免除零
    sums[sums == 0] = 1
    return matrix / sums


def sparse_matrix_to_dense(row_indices: np.ndarray, 
                          col_indices: np.ndarray, 
                          values: np.ndarray, 
                          shape: Tuple[int, int]) -> np.ndarray:
    """
    将稀疏矩阵表示转换为密集矩阵
    
    参数:
    row_indices: 行索引数组
    col_indices: 列索引数组
    values: 对应的值数组
    shape: 输出矩阵的形状
    
    返回:
    np.ndarray: 密集矩阵
    """
    matrix = np.zeros(shape)
    for i, (row, col) in enumerate(zip(row_indices, col_indices)):
        matrix[row, col] = values[i]
    return matrix


def dense_matrix_to_sparse(matrix: np.ndarray, 
                          threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将密集矩阵转换为稀疏表示
    
    参数:
    matrix: 输入密集矩阵
    threshold: 阈值，绝对值小于该值的元素将被视为零
    
    返回:
    Tuple: (行索引, 列索引, 值数组)
    """
    rows, cols = np.where(np.abs(matrix) > threshold)
    values = matrix[rows, cols]
    return rows, cols, values


def calculate_eigenvalues(matrix: np.ndarray, 
                         top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算矩阵的特征值和特征向量
    
    参数:
    matrix: 输入矩阵
    top_k: 只返回最大的k个特征值和对应的特征向量，如果为None则返回全部
    
    返回:
    Tuple: (特征值数组, 特征向量数组)
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    if top_k is not None:
        # 按特征值大小排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 只保留前k个
        eigenvalues = eigenvalues[:top_k]
        eigenvectors = eigenvectors[:, :top_k]
    
    return eigenvalues, eigenvectors


def matrix_power(matrix: np.ndarray, power: int) -> np.ndarray:
    """
    计算矩阵的幂
    
    参数:
    matrix: 输入矩阵
    power: 幂次
    
    返回:
    np.ndarray: 矩阵的幂
    """
    if power < 0:
        raise ValueError("幂次必须为非负数")
    
    if power == 0:
        return np.eye(matrix.shape[0])
    
    result = matrix.copy()
    for _ in range(1, power):
        result = np.dot(result, matrix)
    
    return result


def create_transition_matrix_from_policy(policy: np.ndarray, 
                                        state_transitions: List[List[int]]) -> np.ndarray:
    """
    从策略创建状态转移矩阵
    
    参数:
    policy: 策略矩阵 [S x A]，每行表示一个状态的动作概率
    state_transitions: 列表，每个元素是一个状态在各个动作下可能的下一个状态列表
    
    返回:
    np.ndarray: 状态转移矩阵 [S x S]
    """
    state_size = policy.shape[0]
    action_size = policy.shape[1]
    
    transition_matrix = np.zeros((state_size, state_size))
    
    for s in range(state_size):
        for a in range(action_size):
            a_prob = policy[s, a]
            
            if a_prob > 0 and a < len(state_transitions[s]):
                next_s = state_transitions[s][a]
                transition_matrix[s, next_s] += a_prob
    
    return transition_matrix


def calculate_stationary_distribution(transition_matrix: np.ndarray, 
                                     max_iterations: int = 1000, 
                                     tolerance: float = 1e-8) -> np.ndarray:
    """
    计算马尔可夫链的平稳分布
    
    参数:
    transition_matrix: 状态转移矩阵 [S x S]
    max_iterations: 最大迭代次数
    tolerance: 收敛容差
    
    返回:
    np.ndarray: 平稳分布向量 [S]
    """
    state_size = transition_matrix.shape[0]
    
    # 方法1: 幂迭代法
    distribution = np.ones(state_size) / state_size  # 均匀初始化
    
    for _ in range(max_iterations):
        new_distribution = np.dot(distribution, transition_matrix)
        
        # 检查收敛
        if np.max(np.abs(new_distribution - distribution)) < tolerance:
            return new_distribution / np.sum(new_distribution)  # 归一化
        
        distribution = new_distribution
    
    # 如果没有收敛，返回最后的结果
    return distribution / np.sum(distribution)


def vector_matrix_multiply(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    执行向量-矩阵乘法 (仿真MBdot操作)
    
    参数:
    vec: 输入向量 [1 x N]
    mat: 矩阵 [N x M]
    
    返回:
    np.ndarray: 结果向量 [1 x M]
    """
    return np.dot(vec, mat)


if __name__ == "__main__":
    # 简单测试
    print("矩阵操作测试")
    
    # 创建一个测试矩阵
    test_matrix = np.array([
        [0.1, 0.2, 0.3, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.2, 0.0, 0.3, 0.5],
        [0.1, 0.2, 0.0, 0.7]
    ])
    
    print("\n原始矩阵:")
    print(test_matrix)
    
    print("\n按行归一化后的矩阵:")
    row_normalized = normalize_matrix(test_matrix, axis=1)
    print(row_normalized)
    print("行和:", row_normalized.sum(axis=1))
    
    print("\n按列归一化后的矩阵:")
    col_normalized = normalize_matrix(test_matrix, axis=0)
    print(col_normalized)
    print("列和:", col_normalized.sum(axis=0))
    
    print("\n稀疏矩阵转换:")
    rows, cols, values = dense_matrix_to_sparse(test_matrix, threshold=0.1)
    print(f"非零元素数量: {len(values)}")
    print(f"行索引: {rows}")
    print(f"列索引: {cols}")
    print(f"值: {values}")
    
    print("\n转换回密集矩阵:")
    dense_again = sparse_matrix_to_dense(rows, cols, values, test_matrix.shape)
    print(dense_again)
    
    print("\n矩阵特征值:")
    eigenvalues, eigenvectors = calculate_eigenvalues(test_matrix, top_k=2)
    print(f"前两个特征值: {eigenvalues}")
    
    print("\n平稳分布:")
    # 确保是一个有效的随机矩阵
    valid_trans_matrix = normalize_matrix(test_matrix)
    stationary = calculate_stationary_distribution(valid_trans_matrix)
    print(stationary)
    print("验证: π·P ≈ π")
    print(np.dot(stationary, valid_trans_matrix)) 