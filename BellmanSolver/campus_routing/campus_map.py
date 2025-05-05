"""
校园地图模块

该模块实现了校园地图的表示和操作，包括建筑物、道路和导航功能。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Dict, Optional, Union, Any, Set
import networkx as nx
import random
import math


class Building:
    """校园建筑物类"""
    
    def __init__(self, 
                name: str, 
                position: Tuple[float, float], 
                size: Tuple[float, float],
                color: str = 'skyblue', 
                is_landmark: bool = False):
        """
        初始化建筑物
        
        参数:
        name: 建筑物名称
        position: 建筑物位置 (x, y)
        size: 建筑物大小 (width, height)
        color: 建筑物颜色
        is_landmark: 是否是地标建筑
        """
        self.name = name
        self.position = position
        self.size = size
        self.color = color
        self.is_landmark = is_landmark
        
        # 计算建筑物中心点
        self.center = (position[0] + size[0] / 2, position[1] + size[1] / 2)
        
        # 计算建筑物的边界
        self.boundaries = {
            'left': position[0],
            'right': position[0] + size[0],
            'bottom': position[1],
            'top': position[1] + size[1]
        }
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        检查点是否在建筑物内
        
        参数:
        point: 要检查的点 (x, y)
        
        返回:
        bool: 点是否在建筑物内
        """
        x, y = point
        return (self.boundaries['left'] <= x <= self.boundaries['right'] and
                self.boundaries['bottom'] <= y <= self.boundaries['top'])
    
    def get_nearest_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        获取建筑物边界上最近的点
        
        参数:
        point: 参考点 (x, y)
        
        返回:
        Tuple[float, float]: 建筑物边界上最近的点
        """
        x, y = point
        
        # 计算点到边界的最短距离
        distances = {
            'left': abs(x - self.boundaries['left']),
            'right': abs(x - self.boundaries['right']),
            'bottom': abs(y - self.boundaries['bottom']),
            'top': abs(y - self.boundaries['top'])
        }
        
        # 找到最近的边界
        nearest_boundary = min(distances, key=distances.get)
        
        # 计算边界上的点
        if nearest_boundary == 'left':
            return (self.boundaries['left'], 
                   min(max(y, self.boundaries['bottom']), self.boundaries['top']))
        elif nearest_boundary == 'right':
            return (self.boundaries['right'], 
                   min(max(y, self.boundaries['bottom']), self.boundaries['top']))
        elif nearest_boundary == 'bottom':
            return (min(max(x, self.boundaries['left']), self.boundaries['right']), 
                   self.boundaries['bottom'])
        else:  # top
            return (min(max(x, self.boundaries['left']), self.boundaries['right']), 
                   self.boundaries['top'])
    
    def __str__(self) -> str:
        return f"{self.name} at {self.position}"


class Road:
    """校园道路类"""
    
    def __init__(self, 
                start: Tuple[float, float], 
                end: Tuple[float, float],
                name: Optional[str] = None,
                width: float = 1.0,
                type: str = 'normal',  # normal, main, pedestrian
                bidirectional: bool = True):
        """
        初始化道路
        
        参数:
        start: 起点 (x, y)
        end: 终点 (x, y)
        name: 道路名称
        width: 道路宽度
        type: 道路类型
        bidirectional: 是否双向
        """
        self.start = start
        self.end = end
        self.name = name
        self.width = width
        self.type = type
        self.bidirectional = bidirectional
        
        # 计算道路长度
        self.length = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        
        # 设置道路样式
        if type == 'main':
            self.color = 'slategray'
            self.linestyle = '-'
            self.speed = 2.0  # 相对速度
        elif type == 'pedestrian':
            self.color = 'tan'
            self.linestyle = '--'
            self.speed = 0.8
        else:  # normal
            self.color = 'gray'
            self.linestyle = '-'
            self.speed = 1.0
    
    def get_distance(self, point: Tuple[float, float]) -> float:
        """
        计算点到道路的最短距离
        
        参数:
        point: 要计算的点 (x, y)
        
        返回:
        float: 距离
        """
        x, y = point
        x1, y1 = self.start
        x2, y2 = self.end
        
        # 计算点到线段的最短距离
        # 参考: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        
        # 检查点是否在线段的投影范围内
        dot_product = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1))
        squared_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if dot_product < 0 or dot_product > squared_length:
            # 点在线段延长线上，计算到端点的距离
            dist_to_start = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
            dist_to_end = math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            return min(dist_to_start, dist_to_end)
        
        # 点到线段的垂直距离
        return numerator / denominator
    
    def get_nearest_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        获取道路上离给定点最近的点
        
        参数:
        point: 参考点 (x, y)
        
        返回:
        Tuple[float, float]: 道路上最近的点
        """
        x, y = point
        x1, y1 = self.start
        x2, y2 = self.end
        
        # 计算向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 如果线段长度为0，返回起点
        if dx == 0 and dy == 0:
            return self.start
        
        # 计算投影长度比例
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        
        # 限制t在[0,1]之间，确保点在线段上
        t = max(0, min(1, t))
        
        # 计算投影点
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        
        return (nearest_x, nearest_y)
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name}: {self.start} to {self.end}"
        else:
            return f"Road: {self.start} to {self.end}"


class CampusMap:
    """校园地图类"""
    
    def __init__(self, 
                 name: str, 
                 width: float = 100.0, 
                 height: float = 100.0):
        """
        初始化校园地图
        
        参数:
        name: 校区名称
        width: 地图宽度
        height: 地图高度
        """
        self.name = name
        self.width = width
        self.height = height
        
        # 存储建筑物和道路
        self.buildings: Dict[str, Building] = {}
        self.roads: List[Road] = []
        
        # 创建路网图
        self.graph = nx.Graph()
        
        # 存储兴趣点
        self.points_of_interest: Dict[str, Tuple[float, float]] = {}
        
        # 绘图对象
        self.fig = None
        self.ax = None
    
    def add_building(self, building: Building) -> None:
        """
        添加建筑物
        
        参数:
        building: 建筑物对象
        """
        self.buildings[building.name] = building
    
    def add_road(self, road: Road) -> None:
        """
        添加道路
        
        参数:
        road: 道路对象
        """
        self.roads.append(road)
        
        # 更新路网图
        weight = road.length / road.speed  # 考虑速度的权重
        
        # 添加道路到图中
        self.graph.add_edge(road.start, road.end, weight=weight, road=road)
        
        # 如果是双向道路
        if road.bidirectional:
            self.graph.add_edge(road.end, road.start, weight=weight, road=road)
    
    def add_point_of_interest(self, name: str, position: Tuple[float, float]) -> None:
        """
        添加兴趣点
        
        参数:
        name: 兴趣点名称
        position: 兴趣点位置 (x, y)
        """
        self.points_of_interest[name] = position
    
    def find_nearest_road(self, point: Tuple[float, float]) -> Optional[Road]:
        """
        找到离给定点最近的道路
        
        参数:
        point: 目标点 (x, y)
        
        返回:
        Optional[Road]: 最近的道路
        """
        if not self.roads:
            return None
        
        nearest_road = None
        min_distance = float('inf')
        
        for road in self.roads:
            distance = road.get_distance(point)
            if distance < min_distance:
                min_distance = distance
                nearest_road = road
        
        return nearest_road
    
    def find_nearest_road_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        找到离给定点最近的道路上的点
        
        参数:
        point: 目标点 (x, y)
        
        返回:
        Optional[Tuple[float, float]]: 道路上最近的点
        """
        nearest_road = self.find_nearest_road(point)
        if nearest_road:
            return nearest_road.get_nearest_point(point)
        return None
    
    def find_shortest_path(self, 
                         start: Union[str, Tuple[float, float]], 
                         end: Union[str, Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], float]:
        """
        寻找最短路径
        
        参数:
        start: 起点名称或坐标
        end: 终点名称或坐标
        
        返回:
        Tuple[List[Tuple[float, float]], float]: (路径点列表, 路径长度)
        """
        # 解析起点和终点
        start_point = self._parse_location(start)
        end_point = self._parse_location(end)
        
        if start_point is None or end_point is None:
            return [], 0.0
        
        # 找到最接近道路的点
        start_road_point = self.find_nearest_road_point(start_point)
        end_road_point = self.find_nearest_road_point(end_point)
        
        if start_road_point is None or end_road_point is None:
            return [], 0.0
        
        # 临时添加这些点到图中
        temp_edges = []
        
        # 添加起点到最近道路点的边
        start_weight = math.sqrt((start_point[0] - start_road_point[0]) ** 2 + 
                              (start_point[1] - start_road_point[1]) ** 2)
        self.graph.add_edge(start_point, start_road_point, weight=start_weight)
        temp_edges.append((start_point, start_road_point))
        
        # 添加终点到最近道路点的边
        end_weight = math.sqrt((end_point[0] - end_road_point[0]) ** 2 + 
                            (end_point[1] - end_road_point[1]) ** 2)
        self.graph.add_edge(end_road_point, end_point, weight=end_weight)
        temp_edges.append((end_road_point, end_point))
        
        try:
            # 使用Dijkstra算法寻找最短路径
            path = nx.shortest_path(self.graph, source=start_point, target=end_point, weight='weight')
            path_length = nx.shortest_path_length(self.graph, source=start_point, target=end_point, weight='weight')
        except nx.NetworkXNoPath:
            # 没有找到路径
            path = []
            path_length = 0.0
        except nx.NodeNotFound:
            # 节点不在图中
            path = []
            path_length = 0.0
        
        # 移除临时边
        for edge in temp_edges:
            self.graph.remove_edge(*edge)
        
        return path, path_length
    
    def _parse_location(self, location: Union[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        解析位置参数
        
        参数:
        location: 位置名称或坐标
        
        返回:
        Optional[Tuple[float, float]]: 位置坐标
        """
        if isinstance(location, tuple):
            return location
        elif isinstance(location, str):
            # 检查是否是建筑物名称
            if location in self.buildings:
                return self.buildings[location].center
            # 检查是否是兴趣点名称
            elif location in self.points_of_interest:
                return self.points_of_interest[location]
        
        return None
    
    def draw_map(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        绘制校园地图
        
        参数:
        figsize: 图像大小
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # 设置坐标轴
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        
        # 绘制建筑物
        for building in self.buildings.values():
            rect = patches.Rectangle(
                building.position, 
                building.size[0], 
                building.size[1], 
                linewidth=1, 
                edgecolor='black', 
                facecolor=building.color, 
                alpha=0.7
            )
            self.ax.add_patch(rect)
            
            # 添加建筑物名称
            self.ax.text(
                building.center[0], 
                building.center[1], 
                building.name, 
                ha='center', 
                va='center', 
                fontsize=8
            )
        
        # 绘制道路
        for road in self.roads:
            self.ax.plot(
                [road.start[0], road.end[0]], 
                [road.start[1], road.end[1]], 
                color=road.color, 
                linewidth=road.width * 2, 
                linestyle=road.linestyle,
                alpha=0.8
            )
        
        # 绘制兴趣点
        for name, position in self.points_of_interest.items():
            self.ax.plot(
                position[0], 
                position[1], 
                'ro', 
                markersize=6
            )
            self.ax.text(
                position[0], 
                position[1] + 1, 
                name, 
                ha='center', 
                va='bottom', 
                fontsize=7
            )
        
        self.ax.set_title(f"{self.name} 校区地图")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, linestyle='--', alpha=0.3)
    
    def draw_path(self, path: List[Tuple[float, float]], color: str = 'red') -> None:
        """
        在地图上绘制路径
        
        参数:
        path: 路径点列表
        color: 路径颜色
        """
        if not self.ax:
            self.draw_map()
        
        if path:
            x_coords, y_coords = zip(*path)
            self.ax.plot(x_coords, y_coords, color=color, linewidth=3, marker='o', markersize=4)
            
            # 标记起点和终点
            self.ax.plot(path[0][0], path[0][1], 'go', markersize=8)
            self.ax.plot(path[-1][0], path[-1][1], 'r*', markersize=10)
    
    def show(self) -> None:
        """显示地图"""
        plt.tight_layout()
        plt.show()


def create_qingyuan_campus() -> CampusMap:
    """
    创建安徽大学磬苑校区地图
    
    返回:
    CampusMap: 校区地图
    """
    # 创建地图
    campus = CampusMap("安徽大学磬苑", width=100.0, height=80.0)
    
    # 添加主要建筑物
    buildings = [
        Building("磬苑宾馆", (10, 50), (10, 10), "lightcoral", True),
        Building("图书馆", (30, 40), (15, 15), "skyblue", True),
        Building("行政楼", (55, 55), (10, 10), "khaki"),
        Building("体育馆", (65, 30), (15, 10), "lightgreen"),
        Building("礼堂", (25, 65), (10, 8), "plum"),
        Building("综合楼", (45, 25), (12, 10), "powderblue"),
        Building("科技楼", (75, 15), (10, 12), "palegreen"),
        Building("学生公寓A", (15, 15), (8, 20), "wheat"),
        Building("学生公寓B", (85, 50), (8, 20), "wheat"),
        Building("食堂", (30, 10), (10, 8), "sandybrown"),
        Building("西门", (5, 30), (3, 5), "silver", True),
        Building("东门", (95, 45), (3, 5), "silver", True),
        Building("北门", (50, 75), (5, 3), "silver", True),
        Building("南门", (50, 5), (5, 3), "silver", True),
    ]
    
    for building in buildings:
        campus.add_building(building)
    
    # 添加主要道路
    roads = [
        # 环形主干道
        Road((10, 30), (40, 30), "环路1段", 2.0, "main"),
        Road((40, 30), (70, 30), "环路2段", 2.0, "main"),
        Road((70, 30), (70, 60), "环路3段", 2.0, "main"),
        Road((70, 60), (40, 60), "环路4段", 2.0, "main"),
        Road((40, 60), (10, 60), "环路5段", 2.0, "main"),
        Road((10, 60), (10, 30), "环路6段", 2.0, "main"),
        
        # 南北干道
        Road((50, 5), (50, 30), "南北干道1", 1.5, "main"),
        Road((50, 30), (50, 60), "南北干道2", 1.5, "main"),
        Road((50, 60), (50, 75), "南北干道3", 1.5, "main"),
        
        # 东西干道
        Road((5, 40), (30, 40), "东西干道1", 1.5, "main"),
        Road((30, 40), (70, 40), "东西干道2", 1.5, "main"),
        Road((70, 40), (95, 40), "东西干道3", 1.5, "main"),
        
        # 次要道路
        Road((20, 15), (20, 30), "学生宿舍路", 1.0, "normal"),
        Road((20, 30), (20, 60), "中部竖路", 1.0, "normal"),
        Road((30, 15), (45, 25), "食堂综合楼路", 1.0, "normal"),
        Road((45, 25), (65, 30), "综合楼体育馆路", 1.0, "normal"),
        Road((65, 30), (75, 15), "体育馆科技楼路", 1.0, "normal"),
        Road((30, 40), (30, 60), "图书馆路", 1.0, "normal"),
        Road((55, 55), (70, 60), "行政楼路", 1.0, "normal"),
        Road((25, 60), (25, 65), "礼堂路", 1.0, "normal"),
        Road((85, 50), (85, 60), "学生公寓B路", 1.0, "normal"),
        
        # 人行道
        Road((15, 25), (30, 40), "人行道1", 0.5, "pedestrian"),
        Road((35, 35), (45, 25), "人行道2", 0.5, "pedestrian"),
        Road((40, 45), (55, 55), "人行道3", 0.5, "pedestrian"),
        Road((65, 35), (75, 40), "人行道4", 0.5, "pedestrian"),
        Road((40, 15), (50, 25), "人行道5", 0.5, "pedestrian"),
    ]
    
    for road in roads:
        campus.add_road(road)
    
    # 添加兴趣点
    poi = [
        ("篮球场", (60, 20)),
        ("校医院", (35, 50)),
        ("邮局", (25, 30)),
        ("银行ATM", (40, 10)),
        ("咖啡厅", (55, 35)),
        ("计算机中心", (65, 45)),
        ("创新创业中心", (80, 30))
    ]
    
    for name, pos in poi:
        campus.add_point_of_interest(name, pos)
    
    return campus


if __name__ == "__main__":
    # 简单测试
    print("创建安徽大学磬苑校区地图...")
    campus = create_qingyuan_campus()
    
    # 绘制地图
    campus.draw_map()
    
    # 测试路径规划
    start = "图书馆"
    end = "科技楼"
    print(f"规划从 {start} 到 {end} 的路径...")
    
    path, length = campus.find_shortest_path(start, end)
    
    if path:
        print(f"找到路径，长度: {length:.2f}")
        campus.draw_path(path)
    else:
        print("未找到路径")
    
    # 显示地图
    campus.show()
    
    # 测试随机路径
    random_buildings = random.sample(list(campus.buildings.keys()), 2)
    start = random_buildings[0]
    end = random_buildings[1]
    
    print(f"\n规划从 {start} 到 {end} 的路径...")
    path, length = campus.find_shortest_path(start, end)
    
    if path:
        print(f"找到路径，长度: {length:.2f}")
        
        # 绘制新地图
        campus.draw_map()
        campus.draw_path(path, color='blue')
        campus.show()
    else:
        print("未找到路径") 