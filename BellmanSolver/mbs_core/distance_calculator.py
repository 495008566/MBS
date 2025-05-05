"""
距离计算模块

该模块提供计算状态之间距离的功能，用于MBS算法中的预处理阶段，
帮助将状态分组并确定时间步数。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import heapq


def bfs_distance(transition_matrix: np.ndarray, goal_states: List[int]) -> np.ndarray:
    """
    使用广度优先搜索(BFS)计算所有状态到目标状态的最短距离
    
    参数:
    transition_matrix: 状态转移概率矩阵 [S x S]
    goal_states: 目标状态列表
    
    返回:
    np.ndarray: 到目标的距离数组，不可达状态的距离为无穷大
    """
    state_size = transition_matrix.shape[0]
    distances = np.full(state_size, np.inf)
    
    # 初始化目标状态的距离为0
    for goal in goal_states:
        distances[goal] = 0
    
    # 使用队列进行BFS
    queue = goal_states.copy()
    visited = set(goal_states)
    
    while queue:
        current = queue.pop(0)
        current_distance = distances[current]
        
        # 找到所有可以到达当前状态的前驱状态
        predecessors = np.where(transition_matrix[:, current] > 0)[0]
        
        for pred in predecessors:
            if pred not in visited:
                distances[pred] = current_distance + 1
                visited.add(pred)
                queue.append(pred)
    
    return distances


def dijkstra_distance(transition_matrix: np.ndarray, goal_states: List[int]) -> np.ndarray:
    """
    使用Dijkstra算法计算所有状态到目标状态的最短距离，考虑转移概率
    
    参数:
    transition_matrix: 状态转移概率矩阵 [S x S]
    goal_states: 目标状态列表
    
    返回:
    np.ndarray: 到目标的距离数组，不可达状态的距离为无穷大
    """
    state_size = transition_matrix.shape[0]
    distances = np.full(state_size, np.inf)
    
    # 初始化目标状态的距离为0
    for goal in goal_states:
        distances[goal] = 0
    
    # 使用优先队列进行Dijkstra算法
    # 队列中的元素是(距离, 状态)元组
    priority_queue = [(0, goal) for goal in goal_states]
    heapq.heapify(priority_queue)
    
    while priority_queue:
        current_distance, current_state = heapq.heappop(priority_queue)
        
        # 如果已经找到更短的路径，跳过
        if current_distance > distances[current_state]:
            continue
        
        # 找到所有可以到达当前状态的前驱状态
        predecessors = np.where(transition_matrix[:, current_state] > 0)[0]
        
        for pred in predecessors:
            # 使用转移概率的负对数作为边的权重
            # 概率越高，权重越低
            edge_weight = -np.log(transition_matrix[pred, current_state] + 1e-10)
            distance = current_distance + edge_weight
            
            if distance < distances[pred]:
                distances[pred] = distance
                heapq.heappush(priority_queue, (distance, pred))
    
    return distances


def group_states_by_distance(distances: np.ndarray) -> List[List[int]]:
    """
    按到目标的距离将状态分组
    
    参数:
    distances: 每个状态到目标的距离数组
    
    返回:
    List[List[int]]: 按距离分组的状态列表
    """
    # 找出有限距离的最大值
    finite_distances = distances[np.isfinite(distances)]
    if len(finite_distances) == 0:
        return []
    
    max_distance = int(np.max(finite_distances))
    layers = [[] for _ in range(max_distance + 1)]
    
    for state, dist in enumerate(distances):
        if np.isfinite(dist):
            layers[int(dist)].append(state)
    
    return layers


def find_critical_states(transition_matrix: np.ndarray, 
                        distances: np.ndarray, 
                        threshold: float = 0.1) -> List[int]:
    """
    找出关键状态（从这些状态可以到达多个距离不同的状态）
    
    参数:
    transition_matrix: 状态转移概率矩阵 [S x S]
    distances: 每个状态到目标的距离数组
    threshold: 转移概率阈值
    
    返回:
    List[int]: 关键状态列表
    """
    state_size = transition_matrix.shape[0]
    critical_states = []
    
    for s in range(state_size):
        # 跳过距离无穷大的状态
        if not np.isfinite(distances[s]):
            continue
        
        # 找出所有可从当前状态到达的状态
        successors = np.where(transition_matrix[s, :] > threshold)[0]
        
        # 检查这些后继状态的距离是否不同
        successor_distances = set([distances[succ] for succ in successors if np.isfinite(distances[succ])])
        
        # 如果有多个不同的距离，则是关键状态
        if len(successor_distances) > 1:
            critical_states.append(s)
    
    return critical_states


if __name__ == "__main__":
    # 简单测试
    print("距离计算测试")
    
    # 创建一个简单的5x5网格世界的转移矩阵
    state_size = 25
    transition_matrix = np.zeros((state_size, state_size))
    
    # 添加四个方向的转移
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    
    for s in range(state_size):
        row, col = s // 5, s % 5
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            # 检查是否在网格内
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                new_s = new_row * 5 + new_col
                transition_matrix[s, new_s] = 0.25  # 等概率
    
    # 设置目标状态为右下角
    goal_states = [24]
    
    # 使用BFS计算距离
    bfs_distances = bfs_distance(transition_matrix, goal_states)
    print("\nBFS距离:")
    print(bfs_distances.reshape(5, 5))
    
    # 按距离分组
    groups = group_states_by_distance(bfs_distances)
    print("\n按距离分组的状态:")
    for i, group in enumerate(groups):
        print(f"距离 {i}: {group}")
    
    # 找出关键状态
    critical = find_critical_states(transition_matrix, bfs_distances)
    print("\n关键状态:")
    print(critical)
    
    # 使用Dijkstra计算距离（这里结果应与BFS相同，因为我们使用了等概率转移）
    dijkstra_distances = dijkstra_distance(transition_matrix, goal_states)
    print("\nDijkstra距离:")
    print(dijkstra_distances.reshape(5, 5)) 