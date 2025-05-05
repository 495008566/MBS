"""
状态空间管理器模块

该模块提供状态空间的管理功能，包括状态命名、属性设置和检索、状态分组等。
用于处理各种强化学习环境中的状态表示。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Union


class StateManager:
    """
    状态空间管理器，负责状态映射和操作
    """
    
    def __init__(self, state_size: int):
        """
        初始化状态管理器
        
        参数:
        state_size: 状态空间大小
        """
        self.state_size = state_size
        self.state_names = {i: f"State_{i}" for i in range(state_size)}
        self.state_properties = {}
        self.state_groups = {}
    
    def set_state_name(self, state_id: int, name: str) -> None:
        """
        设置状态名称
        
        参数:
        state_id: 状态ID
        name: 状态名称
        """
        if 0 <= state_id < self.state_size:
            self.state_names[state_id] = name
        else:
            raise ValueError(f"状态ID {state_id} 超出范围 [0, {self.state_size-1}]")
    
    def set_state_property(self, state_id: int, property_name: str, value: Any) -> None:
        """
        设置状态属性
        
        参数:
        state_id: 状态ID
        property_name: 属性名称
        value: 属性值
        """
        if 0 <= state_id < self.state_size:
            if state_id not in self.state_properties:
                self.state_properties[state_id] = {}
            self.state_properties[state_id][property_name] = value
        else:
            raise ValueError(f"状态ID {state_id} 超出范围 [0, {self.state_size-1}]")
    
    def set_multiple_state_property(self, state_ids: List[int], property_name: str, value: Any) -> None:
        """
        为多个状态设置相同的属性
        
        参数:
        state_ids: 状态ID列表
        property_name: 属性名称
        value: 属性值
        """
        for state_id in state_ids:
            self.set_state_property(state_id, property_name, value)
    
    def get_state_property(self, state_id: int, property_name: str, default: Any = None) -> Any:
        """
        获取状态属性
        
        参数:
        state_id: 状态ID
        property_name: 属性名称
        default: 如果属性不存在，返回的默认值
        
        返回:
        Any: 属性值或默认值
        """
        if 0 <= state_id < self.state_size:
            if state_id in self.state_properties:
                return self.state_properties[state_id].get(property_name, default)
        return default
    
    def get_states_by_property(self, property_name: str, value: Any) -> List[int]:
        """
        获取具有特定属性值的所有状态
        
        参数:
        property_name: 属性名称
        value: 属性值
        
        返回:
        List[int]: 满足条件的状态ID列表
        """
        result = []
        for state_id in range(self.state_size):
            if self.get_state_property(state_id, property_name) == value:
                result.append(state_id)
        return result
    
    def add_state_to_group(self, group_name: str, state_id: int) -> None:
        """
        将状态添加到组
        
        参数:
        group_name: 组名称
        state_id: 状态ID
        """
        if 0 <= state_id < self.state_size:
            if group_name not in self.state_groups:
                self.state_groups[group_name] = set()
            self.state_groups[group_name].add(state_id)
        else:
            raise ValueError(f"状态ID {state_id} 超出范围 [0, {self.state_size-1}]")
    
    def remove_state_from_group(self, group_name: str, state_id: int) -> None:
        """
        从组中移除状态
        
        参数:
        group_name: 组名称
        state_id: 状态ID
        """
        if group_name in self.state_groups and state_id in self.state_groups[group_name]:
            self.state_groups[group_name].remove(state_id)
    
    def get_group_states(self, group_name: str) -> List[int]:
        """
        获取组中的所有状态
        
        参数:
        group_name: 组名称
        
        返回:
        List[int]: 组中的状态ID列表
        """
        if group_name in self.state_groups:
            return sorted(list(self.state_groups[group_name]))
        return []
    
    def get_goal_states(self) -> List[int]:
        """
        获取目标状态列表
        
        返回:
        List[int]: 目标状态ID列表
        """
        return self.get_states_by_property("is_goal", True)
    
    def get_state_name(self, state_id: int) -> str:
        """
        获取状态名称
        
        参数:
        state_id: 状态ID
        
        返回:
        str: 状态名称
        """
        if 0 <= state_id < self.state_size:
            return self.state_names.get(state_id, f"State_{state_id}")
        return f"Invalid_State_{state_id}"
    
    def create_distance_layers(self, goal_states: Optional[List[int]] = None) -> List[List[int]]:
        """
        根据到目标状态的距离创建分层状态列表
        
        参数:
        goal_states: 目标状态列表，如果为None则使用is_goal属性为True的状态
        
        返回:
        List[List[int]]: 分层的状态列表，每层包含距离相同的状态
        """
        if goal_states is None:
            goal_states = self.get_goal_states()
            
        if not goal_states:
            raise ValueError("未指定目标状态且找不到有is_goal=True属性的状态")
        
        # 使用BFS计算距离
        distances = [-1] * self.state_size
        for goal in goal_states:
            distances[goal] = 0
        
        queue = list(goal_states)
        
        # 构建反向连接: 谁可以到达当前状态
        transitions = self.get_state_property(0, "transitions", None)
        if transitions is None:
            raise ValueError("状态需要设置transitions属性以计算距离")
        
        reverse_transitions = [[] for _ in range(self.state_size)]
        for s in range(self.state_size):
            s_transitions = self.get_state_property(s, "transitions", [])
            for next_s in s_transitions:
                reverse_transitions[next_s].append(s)
        
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            
            # 查找可以到达当前状态的所有状态
            for prev_s in reverse_transitions[current]:
                if distances[prev_s] == -1:  # 尚未访问
                    distances[prev_s] = current_dist + 1
                    queue.append(prev_s)
        
        # 创建分层结构
        max_dist = max([d for d in distances if d != -1], default=0)
        layers = [[] for _ in range(max_dist + 1)]
        
        for s, dist in enumerate(distances):
            if dist != -1:  # 只包含可达状态
                layers[dist].append(s)
        
        return layers


if __name__ == "__main__":
    # 简单测试
    manager = StateManager(25)  # 5x5网格世界
    
    # 设置一些状态属性
    manager.set_state_name(24, "Goal")
    manager.set_state_property(24, "is_goal", True)
    manager.set_state_property(24, "reward", 10.0)
    
    manager.set_state_name(12, "Trap1")
    manager.set_state_property(12, "is_trap", True)
    manager.set_state_property(12, "reward", -10.0)
    
    manager.set_state_name(18, "Trap2")
    manager.set_state_property(18, "is_trap", True)
    manager.set_state_property(18, "reward", -5.0)
    
    # 设置过渡概率(只是示例，不是一个有效的网格)
    for s in range(25):
        row, col = s // 5, s % 5
        transitions = []
        
        # 上
        if row > 0:
            transitions.append((row-1) * 5 + col)
        
        # 下
        if row < 4:
            transitions.append((row+1) * 5 + col)
            
        # 左
        if col > 0:
            transitions.append(row * 5 + col-1)
            
        # 右
        if col < 4:
            transitions.append(row * 5 + col+1)
            
        manager.set_state_property(s, "transitions", transitions)
    
    # 测试分组
    manager.add_state_to_group("corners", 0)  # 左上
    manager.add_state_to_group("corners", 4)  # 右上
    manager.add_state_to_group("corners", 20)  # 左下
    manager.add_state_to_group("corners", 24)  # 右下
    
    print("角落状态:", manager.get_group_states("corners"))
    print("目标状态:", manager.get_goal_states())
    print("陷阱状态:", manager.get_states_by_property("is_trap", True))
    
    # 这部分会失败，因为我们没有设置完整的转移信息
    try:
        distance_layers = manager.create_distance_layers()
        print("距离分层:", distance_layers)
    except ValueError as e:
        print(f"预期的错误: {e}") 