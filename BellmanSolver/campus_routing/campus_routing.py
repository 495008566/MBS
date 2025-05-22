"""
校园路径规划模块

该模块使用贝尔曼方程进行校园路径规划，可以求解最短路径、最快路径或其他自定义权重的路径。
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union, Any, Set, Callable
import random
import math
import time

from .campus_map import CampusMap, Building, Road, create_qingyuan_campus


class CampusRouter:
    """校园路径规划器"""
    
    def __init__(self, campus_map: CampusMap):
        """
        初始化路径规划器
        
        参数:
        campus_map: 校园地图
        """
        self.campus_map = campus_map
        
        # 提取路网图
        self.graph = campus_map.graph.copy()
        
        # 节点索引映射：坐标 -> 索引
        self.node_to_index = {}
        
        # 索引映射：索引 -> 坐标
        self.index_to_node = {}
        
        # 创建索引映射
        self._create_index_mappings()
        
        # 转移概率矩阵
        self.transition_matrix = None
        
        # 奖励向量
        self.reward_vector = None
        
        # 保存最新计算结果
        self.last_value_function = None
        self.last_policy = None
        self.last_path = None
    
    def _create_index_mappings(self) -> None:
        """创建节点坐标和索引之间的映射"""
        # 获取图中所有节点
        nodes = list(self.graph.nodes())
        
        # 创建双向映射
        for i, node in enumerate(nodes):
            self.node_to_index[node] = i
            self.index_to_node[i] = node
    
    def build_matrices(self, 
                      destination: Union[str, Tuple[float, float]], 
                      distance_weight: float = 1.0,
                      time_weight: float = 0.0,
                      custom_weights: Optional[Dict[Tuple[float, float], float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建转移概率矩阵和奖励向量
        
        参数:
        destination: 目标位置名称或坐标
        distance_weight: 距离权重
        time_weight: 时间权重
        custom_weights: 自定义节点权重
        
        返回:
        Tuple[np.ndarray, np.ndarray]: (转移矩阵, 奖励向量)
        """
        # 解析目标位置
        dest_point = self.campus_map._parse_location(destination)
        if dest_point is None:
            raise ValueError(f"无法解析目标位置: {destination}")
        
        # 将目标点映射到最近的道路点
        dest_road_point = self.campus_map.find_nearest_road_point(dest_point)
        if dest_road_point is None:
            raise ValueError(f"目标点附近没有道路: {dest_point}")
        
        # 获取目标点的索引
        if dest_road_point in self.node_to_index:
            dest_index = self.node_to_index[dest_road_point]
        else:
            # 如果目标点不在图中，添加到图中
            for node in self.graph.nodes():
                distance = math.sqrt((node[0] - dest_road_point[0])**2 + (node[1] - dest_road_point[1])**2)
                self.graph.add_edge(node, dest_road_point, weight=distance)
            
            # 更新索引映射
            self._create_index_mappings()
            dest_index = self.node_to_index[dest_road_point]
        
        # 节点数量
        n = len(self.node_to_index)
        
        # 创建转移概率矩阵 P[s][s']
        P = np.zeros((n, n))
        
        # 创建奖励向量 R[s]
        R = np.ones(n) * -0.01  # 默认奖励为小负数，鼓励找到更短的路径
        
        # 设置目标状态的奖励
        R[dest_index] = 1.0
        
        # 设置目标状态的转移：自循环
        P[dest_index, dest_index] = 1.0
        
        # 设置其他状态的转移概率和奖励
        for node, idx in self.node_to_index.items():
            if idx == dest_index:
                continue  # 跳过目标状态
            
            # 获取当前节点的所有邻居
            neighbors = list(self.graph.neighbors(node))
            
            if not neighbors:
                # 孤立节点，自循环
                P[idx, idx] = 1.0
                continue
            
            # 计算到每个邻居的权重
            weights = []
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(node, neighbor)
                distance = edge_data.get('weight', 1.0)
                
                # 如果有道路对象，考虑速度
                road = edge_data.get('road')
                time_factor = 1.0 / road.speed if road else 1.0
                
                # 计算总权重
                weight = distance_weight * distance + time_weight * time_factor
                
                # 应用自定义权重
                if custom_weights and neighbor in custom_weights:
                    weight += custom_weights[neighbor]
                
                # 确保权重为正数（贝尔曼方程中需要）
                weights.append(max(0.001, weight))
            
            # 归一化权重，转换为概率（反比例，权重越小，概率越大）
            inv_weights = [1.0 / w for w in weights]
            total = sum(inv_weights)
            probs = [w / total for w in inv_weights]
            
            # 设置转移概率
            for i, neighbor in enumerate(neighbors):
                neigh_idx = self.node_to_index[neighbor]
                P[idx, neigh_idx] = probs[i]
        
        # 保存矩阵和向量以供后续使用
        self.transition_matrix = P
        self.reward_vector = R
        
        return P, R
    
    def value_iteration(self, 
                       P: np.ndarray, 
                       R: np.ndarray, 
                       gamma: float = 0.9, 
                       epsilon: float = 1e-6, 
                       max_iter: int = 1000) -> np.ndarray:
        """
        值迭代算法求解价值函数
        
        参数:
        P: 转移概率矩阵
        R: 奖励向量
        gamma: 折扣因子
        epsilon: 收敛阈值
        max_iter: 最大迭代次数
        
        返回:
        np.ndarray: 值函数
        """
        n = len(R)
        V = np.zeros(n)
        
        for i in range(max_iter):
            V_new = R + gamma * np.dot(P, V)
            
            # 检查收敛
            if np.max(np.abs(V_new - V)) < epsilon:
                print(f"值迭代在第 {i+1} 次迭代后收敛")
                break
            
            V = V_new
        
        # 保存最新值函数
        self.last_value_function = V
        
        return V
    
    def extract_policy(self, V: np.ndarray, P: np.ndarray, gamma: float = 0.9) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """
        从值函数中提取策略
        
        参数:
        V: 值函数
        P: 转移概率矩阵
        gamma: 折扣因子
        
        返回:
        Dict[Tuple[float, float], Tuple[float, float]]: 策略（当前位置 -> 下一个位置）
        """
        n = len(V)
        policy = {}
        
        for node, idx in self.node_to_index.items():
            # 获取所有邻居
            neighbors = list(self.graph.neighbors(node))
            
            if not neighbors:
                continue  # 跳过孤立节点
            
            # 选择价值最高的下一个状态
            best_next_idx = -1
            best_value = float('-inf')
            
            for neighbor in neighbors:
                neigh_idx = self.node_to_index[neighbor]
                expected_value = self.reward_vector[idx] + gamma * V[neigh_idx]
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_next_idx = neigh_idx
            
            if best_next_idx != -1:
                best_next_node = self.index_to_node[best_next_idx]
                policy[node] = best_next_node
        
        # 保存最新策略
        self.last_policy = policy
        
        return policy
    
    def find_path(self, 
                 start: Union[str, Tuple[float, float]], 
                 destination: Union[str, Tuple[float, float]], 
                 distance_weight: float = 1.0,
                 time_weight: float = 0.0,
                 custom_weights: Optional[Dict[Tuple[float, float], float]] = None,
                 max_steps: int = 100) -> List[Tuple[float, float]]:
        """
        寻找从起点到终点的路径
        
        参数:
        start: 起点名称或坐标
        destination: 终点名称或坐标
        distance_weight: 距离权重
        time_weight: 时间权重
        custom_weights: 自定义节点权重
        max_steps: 最大步数
        
        返回:
        List[Tuple[float, float]]: 路径点列表
        """
        # 解析起点和终点
        start_point = self.campus_map._parse_location(start)
        dest_point = self.campus_map._parse_location(destination)
        
        if start_point is None or dest_point is None:
            raise ValueError(f"无法解析起点或终点: {start} -> {destination}")
        
        # 将点映射到最近的道路点
        start_road_point = self.campus_map.find_nearest_road_point(start_point)
        dest_road_point = self.campus_map.find_nearest_road_point(dest_point)
        
        if start_road_point is None or dest_road_point is None:
            raise ValueError("无法找到起点或终点附近的道路")
        
        # 构建转移矩阵和奖励向量
        P, R = self.build_matrices(dest_road_point, distance_weight, time_weight, custom_weights)
        
        # 值迭代求解
        V = self.value_iteration(P, R)
        
        # 提取策略
        policy = self.extract_policy(V, P)
        
        # 使用策略构建路径
        path = [start_point]  # 起始于原始起点
        
        # 添加起点到起始道路点的路径
        if start_point != start_road_point:
            path.append(start_road_point)
        
        # 从起始道路点开始遵循策略
        current_node = start_road_point
        
        for _ in range(max_steps):
            if current_node not in policy:
                break
            
            next_node = policy[current_node]
            path.append(next_node)
            
            # 检查是否到达终点附近
            if (math.sqrt((next_node[0] - dest_road_point[0])**2 + 
                       (next_node[1] - dest_road_point[1])**2) < 0.1):
                break
            
            current_node = next_node
        
        # 添加终点道路点到原始终点的路径
        if path[-1] != dest_road_point:
            path.append(dest_road_point)
        
        if path[-1] != dest_point:
            path.append(dest_point)
        
        # 保存最新路径
        self.last_path = path
        
        return path
    
    def visualize_value_function(self, title: str = "值函数可视化") -> None:
        """
        可视化最近计算的值函数
        
        参数:
        title: 图表标题
        """
        if self.last_value_function is None:
            print("请先运行寻路算法计算值函数")
            return
        
        # 创建散点图
        plt.figure(figsize=(10, 8))
        
        # 获取所有节点坐标和值
        x = []
        y = []
        values = []
        
        for idx, value in enumerate(self.last_value_function):
            node = self.index_to_node[idx]
            x.append(node[0])
            y.append(node[1])
            values.append(value)
        
        # 绘制散点图，颜色表示值大小
        scatter = plt.scatter(x, y, c=values, cmap='viridis', s=50, alpha=0.8)
        
        # 添加颜色条
        plt.colorbar(scatter, label='值函数')
        
        # 添加背景校园地图
        self._draw_background_map()
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_policy(self, title: str = "策略可视化") -> None:
        """
        可视化最近计算的策略
        
        参数:
        title: 图表标题
        """
        if self.last_policy is None:
            print("请先运行寻路算法计算策略")
            return
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 添加背景校园地图
        self._draw_background_map()
        
        # 绘制策略箭头
        for node, next_node in self.last_policy.items():
            plt.arrow(node[0], node[1], 
                     next_node[0] - node[0], next_node[1] - node[1],
                     head_width=1.0, head_length=1.5, fc='blue', ec='blue',
                     length_includes_head=True, alpha=0.6)
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_path(self, path: Optional[List[Tuple[float, float]]] = None, title: str = "路径可视化") -> None:
        """
        可视化路径
        
        参数:
        path: 路径点列表，如果为None则使用最近计算的路径
        title: 图表标题
        """
        if path is None:
            path = self.last_path
        
        if path is None:
            print("请先运行寻路算法或提供路径")
            return
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 添加背景校园地图
        self._draw_background_map()
        
        # 绘制路径
        x, y = zip(*path)
        plt.plot(x, y, 'r-', linewidth=3, alpha=0.7)
        
        # 标记起点和终点
        plt.plot(path[0][0], path[0][1], 'go', markersize=10, label='起点')
        plt.plot(path[-1][0], path[-1][1], 'r*', markersize=12, label='终点')
        
        # 绘制路径点
        plt.plot(x, y, 'ro', markersize=4)
        
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _draw_background_map(self) -> None:
        """绘制背景校园地图"""
        # 绘制建筑物
        for building in self.campus_map.buildings.values():
            rect = plt.Rectangle(
                building.position, 
                building.size[0], 
                building.size[1], 
                linewidth=1, 
                edgecolor='black', 
                facecolor=building.color, 
                alpha=0.3
            )
            plt.gca().add_patch(rect)
        
        # 绘制道路
        for road in self.campus_map.roads:
            plt.plot(
                [road.start[0], road.end[0]], 
                [road.start[1], road.end[1]], 
                color=road.color, 
                linewidth=road.width * 1.5, 
                linestyle=road.linestyle,
                alpha=0.5
            )
        
        # 设置坐标轴
        plt.xlim(0, self.campus_map.width)
        plt.ylim(0, self.campus_map.height)
        plt.grid(True, linestyle='--', alpha=0.3)
    
    def compare_paths(self, 
                    start: Union[str, Tuple[float, float]], 
                    destination: Union[str, Tuple[float, float]]) -> None:
        """
        比较不同权重的路径
        
        参数:
        start: 起点名称或坐标
        destination: 终点名称或坐标
        """
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 添加背景校园地图
        self._draw_background_map()
        
        # 计算基于距离的路径
        distance_path = self.find_path(start, destination, distance_weight=1.0, time_weight=0.0)
        
        # 计算基于时间的路径
        time_path = self.find_path(start, destination, distance_weight=0.2, time_weight=0.8)
        
        # 计算平衡的路径
        balanced_path = self.find_path(start, destination, distance_weight=0.5, time_weight=0.5)
        
        # 绘制不同的路径
        x_dist, y_dist = zip(*distance_path)
        plt.plot(x_dist, y_dist, 'r-', linewidth=3, alpha=0.7, label='最短距离')
        
        x_time, y_time = zip(*time_path)
        plt.plot(x_time, y_time, 'g-', linewidth=3, alpha=0.7, label='最短时间')
        
        x_bal, y_bal = zip(*balanced_path)
        plt.plot(x_bal, y_bal, 'b-', linewidth=3, alpha=0.7, label='平衡路径')
        
        # 标记起点和终点
        start_point = self.campus_map._parse_location(start)
        dest_point = self.campus_map._parse_location(destination)
        
        plt.plot(start_point[0], start_point[1], 'ko', markersize=10, label='起点')
        plt.plot(dest_point[0], dest_point[1], 'k*', markersize=12, label='终点')
        
        # 添加标题和图例
        plt.title(f"从 {start} 到 {destination} 的不同路径比较")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 创建校园地图
    campus = create_qingyuan_campus()
    
    # 创建路径规划器
    router = CampusRouter(campus)
    
    # 测试路径规划
    start = "图书馆"
    end = "科技楼"
    print(f"使用贝尔曼方程规划从 {start} 到 {end} 的路径...")
    
    path = router.find_path(start, end)
    
    # 可视化值函数
    router.visualize_value_function("图书馆到科技楼的值函数")
    
    # 可视化策略
    router.visualize_policy("图书馆到科技楼的策略")
    
    # 可视化路径
    router.visualize_path(path, "图书馆到科技楼的路径")
    
    # 比较不同的路径
    router.compare_paths("图书馆", "科技楼")
    
    
    # 随机测试
    buildings = list(campus.buildings.keys())
    random_start = random.choice(buildings)
    random_end = random.choice([b for b in buildings if b != random_start])
    
    print(f"\n规划从 {random_start} 到 {random_end} 的随机路径...")
    random_path = router.find_path(random_start, random_end)
    router.visualize_path(random_path, f"从 {random_start} 到 {random_end} 的路径") 
