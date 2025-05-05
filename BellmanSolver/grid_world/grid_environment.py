"""
网格世界环境模块

该模块实现了基本的网格世界环境，包括状态转移、奖励计算和环境可视化。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union, Any
import random


class GridEnvironment:
    """基本的网格世界环境"""
    
    # 定义动作
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, 
                height: int = 5, 
                width: int = 5, 
                start_pos: Tuple[int, int] = (0, 0),
                goal_pos: Optional[Tuple[int, int]] = None,
                obstacles: Optional[List[Tuple[int, int]]] = None,
                traps: Optional[List[Tuple[int, int]]] = None,
                default_reward: float = -0.04,
                goal_reward: float = 1.0,
                trap_reward: float = -1.0,
                obstacle_reward: float = -0.1,
                noise_prob: float = 0.0):
        """
        初始化网格世界环境
        
        参数:
        height: 网格高度
        width: 网格宽度
        start_pos: 起始位置 (行, 列)
        goal_pos: 目标位置 (行, 列)，如果为None则默认为右下角
        obstacles: 障碍物位置列表 [(行, 列), ...]
        traps: 陷阱位置列表 [(行, 列), ...]
        default_reward: 默认移动奖励
        goal_reward: 到达目标奖励
        trap_reward: 陷阱奖励
        obstacle_reward: 撞到障碍物的奖励
        noise_prob: 动作随机性概率 (0.0表示确定性环境)
        """
        self.height = height
        self.width = width
        self.start_pos = start_pos
        
        # 如果未指定目标位置，默认为右下角
        self.goal_pos = goal_pos if goal_pos is not None else (height - 1, width - 1)
        
        # 初始化障碍物和陷阱
        self.obstacles = obstacles if obstacles is not None else []
        self.traps = traps if traps is not None else []
        
        # 设置奖励
        self.default_reward = default_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.obstacle_reward = obstacle_reward
        
        # 动作随机性
        self.noise_prob = noise_prob
        
        # 当前位置
        self.current_pos = start_pos
        
        # 终止状态
        self.terminal_states = [self.goal_pos] + self.traps
        
        # 构建状态转移矩阵和奖励矩阵
        self.num_states = height * width
        self.num_actions = 4  # 上右下左
        
        # 状态转移概率矩阵 P[s,a,s'] = 概率
        self.transition_probs = self._build_transition_probs()
        
        # 奖励矩阵 R[s,a,s'] = 奖励
        self.rewards = self._build_rewards()
        
        # 状态值和策略
        self.state_values = np.zeros((height, width))
        self.policy = np.zeros((height, width), dtype=int)
        
        # 轨迹记录
        self.trajectory = []
    
    def _build_transition_probs(self) -> np.ndarray:
        """
        构建状态转移概率矩阵
        
        返回:
        np.ndarray: 转移概率矩阵 [num_states, num_actions, num_states]
        """
        P = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        # 遍历所有状态和动作
        for s in range(self.num_states):
            row, col = self._state_to_pos(s)
            
            # 如果是终止状态，停留在原地
            if (row, col) in self.terminal_states:
                for a in range(self.num_actions):
                    P[s, a, s] = 1.0
                continue
            
            # 如果是障碍物，无法从该状态出发
            if (row, col) in self.obstacles:
                for a in range(self.num_actions):
                    P[s, a, s] = 1.0
                continue
            
            # 计算每个动作的下一个状态
            for a in range(self.num_actions):
                intended_next_pos = self._get_next_pos((row, col), a)
                intended_next_state = self._pos_to_state(intended_next_pos)
                
                # 考虑动作随机性
                if self.noise_prob > 0:
                    # 主方向概率
                    P[s, a, intended_next_state] += 1.0 - self.noise_prob
                    
                    # 其他方向的概率
                    for other_a in range(self.num_actions):
                        if other_a != a:
                            other_next_pos = self._get_next_pos((row, col), other_a)
                            other_next_state = self._pos_to_state(other_next_pos)
                            P[s, a, other_next_state] += self.noise_prob / 3.0
                else:
                    # 确定性环境
                    P[s, a, intended_next_state] = 1.0
        
        return P
    
    def _build_rewards(self) -> np.ndarray:
        """
        构建奖励矩阵
        
        返回:
        np.ndarray: 奖励矩阵 [num_states, num_actions, num_states]
        """
        R = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        # 默认奖励
        R.fill(self.default_reward)
        
        # 设置目标奖励
        goal_state = self._pos_to_state(self.goal_pos)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                R[s, a, goal_state] = self.goal_reward
        
        # 设置陷阱奖励
        for trap_pos in self.traps:
            trap_state = self._pos_to_state(trap_pos)
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    R[s, a, trap_state] = self.trap_reward
        
        # 设置障碍物奖励
        for obstacle_pos in self.obstacles:
            obstacle_state = self._pos_to_state(obstacle_pos)
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    row, col = self._state_to_pos(s)
                    next_pos = self._get_next_pos((row, col), a)
                    if next_pos == obstacle_pos:
                        # 如果动作会导致撞到障碍物，给予奖励但不改变状态
                        R[s, a, s] = self.obstacle_reward
        
        return R
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """
        将状态索引转换为位置坐标
        
        参数:
        state: 状态索引
        
        返回:
        Tuple[int, int]: 位置坐标 (行, 列)
        """
        row = state // self.width
        col = state % self.width
        return (row, col)
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """
        将位置坐标转换为状态索引
        
        参数:
        pos: 位置坐标 (行, 列)
        
        返回:
        int: 状态索引
        """
        row, col = pos
        # 确保位置在网格内
        row = max(0, min(row, self.height - 1))
        col = max(0, min(col, self.width - 1))
        
        # 如果是障碍物，找到最近的非障碍物位置
        if (row, col) in self.obstacles:
            # 简单处理：保持原位
            row, col = self.current_pos
        
        return row * self.width + col
    
    def _get_next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        获取执行动作后的下一个位置
        
        参数:
        pos: 当前位置 (行, 列)
        action: 动作
        
        返回:
        Tuple[int, int]: 下一个位置 (行, 列)
        """
        row, col = pos
        
        if action == self.UP:
            row = max(0, row - 1)
        elif action == self.RIGHT:
            col = min(self.width - 1, col + 1)
        elif action == self.DOWN:
            row = min(self.height - 1, row + 1)
        elif action == self.LEFT:
            col = max(0, col - 1)
        
        # 检查是否是障碍物
        if (row, col) in self.obstacles:
            # 如果是障碍物，保持原位
            return pos
        
        return (row, col)
    
    def reset(self) -> int:
        """
        重置环境
        
        返回:
        int: 初始状态索引
        """
        self.current_pos = self.start_pos
        self.trajectory = [self.current_pos]
        return self._pos_to_state(self.current_pos)
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        执行动作并转换到下一个状态
        
        参数:
        action: 动作
        
        返回:
        Tuple[int, float, bool]: (下一个状态索引, 奖励, 是否终止)
        """
        current_state = self._pos_to_state(self.current_pos)
        
        # 考虑噪声
        if random.random() < self.noise_prob:
            # 以一定概率随机选择其他动作
            action = random.choice([a for a in range(self.num_actions) if a != action])
        
        # 获取下一个位置
        next_pos = self._get_next_pos(self.current_pos, action)
        next_state = self._pos_to_state(next_pos)
        
        # 获取奖励
        reward = self.rewards[current_state, action, next_state]
        
        # 更新当前位置
        self.current_pos = next_pos
        self.trajectory.append(self.current_pos)
        
        # 检查是否终止
        done = next_pos in self.terminal_states
        
        return next_state, reward, done
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        获取状态转移矩阵
        
        返回:
        np.ndarray: 转移矩阵 [num_states, num_states]
        """
        # 使用当前策略构建转移矩阵
        P = np.zeros((self.num_states, self.num_states))
        
        for s in range(self.num_states):
            row, col = self._state_to_pos(s)
            a = self.policy[row, col]
            P[s, :] = self.transition_probs[s, a, :]
        
        return P
    
    def get_reward_vector(self) -> np.ndarray:
        """
        获取奖励向量
        
        返回:
        np.ndarray: 奖励向量 [num_states]
        """
        R = np.zeros(self.num_states)
        
        for s in range(self.num_states):
            row, col = self._state_to_pos(s)
            a = self.policy[row, col]
            # 计算期望奖励
            for next_s in range(self.num_states):
                R[s] += self.transition_probs[s, a, next_s] * self.rewards[s, a, next_s]
        
        return R
    
    def set_policy(self, policy: np.ndarray) -> None:
        """
        设置策略
        
        参数:
        policy: 策略矩阵 [height, width]
        """
        self.policy = policy
    
    def set_state_values(self, values: np.ndarray) -> None:
        """
        设置状态值
        
        参数:
        values: 状态值矩阵 [height, width]
        """
        self.state_values = values
    
    def render(self, mode: str = 'human') -> Optional[plt.Figure]:
        """
        渲染环境
        
        参数:
        mode: 渲染模式
        
        返回:
        Optional[plt.Figure]: matplotlib图形对象
        """
        if mode == 'human':
            plt.figure(figsize=(8, 8))
            plt.imshow(self.state_values, cmap='coolwarm', interpolation='nearest')
            
            # 添加网格线
            plt.grid(color='black', linestyle='-', linewidth=1)
            plt.xticks(np.arange(-.5, self.width, 1), [])
            plt.yticks(np.arange(-.5, self.height, 1), [])
            
            # 显示障碍物
            for obs in self.obstacles:
                row, col = obs
                plt.text(col, row, 'X', ha='center', va='center', fontsize=20)
            
            # 显示陷阱
            for trap in self.traps:
                row, col = trap
                plt.text(col, row, 'T', ha='center', va='center', fontsize=20)
            
            # 显示目标
            row, col = self.goal_pos
            plt.text(col, row, 'G', ha='center', va='center', fontsize=20)
            
            # 显示当前位置
            row, col = self.current_pos
            plt.plot(col, row, 'o', markersize=10, color='black')
            
            # 显示轨迹
            if len(self.trajectory) > 1:
                traj_rows, traj_cols = zip(*self.trajectory)
                plt.plot(traj_cols, traj_rows, 'o-', markersize=5, color='green')
            
            # 显示策略
            for row in range(self.height):
                for col in range(self.width):
                    action = self.policy[row, col]
                    if (row, col) in self.obstacles:
                        continue
                    
                    if action == self.UP:
                        plt.arrow(col, row, 0, -0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif action == self.RIGHT:
                        plt.arrow(col, row, 0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif action == self.DOWN:
                        plt.arrow(col, row, 0, 0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif action == self.LEFT:
                        plt.arrow(col, row, -0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
            
            plt.colorbar(label='State Value')
            plt.title('Grid World')
            plt.show()
            return None
        elif mode == 'rgb_array':
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(self.state_values, cmap='coolwarm', interpolation='nearest')
            
            # 添加网格线
            plt.grid(color='black', linestyle='-', linewidth=1)
            plt.xticks(np.arange(-.5, self.width, 1), [])
            plt.yticks(np.arange(-.5, self.height, 1), [])
            
            # 显示障碍物
            for obs in self.obstacles:
                row, col = obs
                plt.text(col, row, 'X', ha='center', va='center', fontsize=20)
            
            # 显示陷阱
            for trap in self.traps:
                row, col = trap
                plt.text(col, row, 'T', ha='center', va='center', fontsize=20)
            
            # 显示目标
            row, col = self.goal_pos
            plt.text(col, row, 'G', ha='center', va='center', fontsize=20)
            
            # 显示当前位置
            row, col = self.current_pos
            plt.plot(col, row, 'o', markersize=10, color='black')
            
            # 显示轨迹
            if len(self.trajectory) > 1:
                traj_rows, traj_cols = zip(*self.trajectory)
                plt.plot(traj_cols, traj_rows, 'o-', markersize=5, color='green')
            
            # 显示策略
            for row in range(self.height):
                for col in range(self.width):
                    action = self.policy[row, col]
                    if (row, col) in self.obstacles:
                        continue
                    
                    if action == self.UP:
                        plt.arrow(col, row, 0, -0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif action == self.RIGHT:
                        plt.arrow(col, row, 0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif action == self.DOWN:
                        plt.arrow(col, row, 0, 0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif action == self.LEFT:
                        plt.arrow(col, row, -0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
            
            plt.colorbar(label='State Value')
            plt.title('Grid World')
            
            return fig


class StochasticGridEnvironment(GridEnvironment):
    """具有随机转移的网格世界环境"""
    
    def __init__(self, 
                height: int = 5, 
                width: int = 5, 
                start_pos: Tuple[int, int] = (0, 0),
                goal_pos: Optional[Tuple[int, int]] = None,
                obstacles: Optional[List[Tuple[int, int]]] = None,
                traps: Optional[List[Tuple[int, int]]] = None,
                default_reward: float = -0.04,
                goal_reward: float = 1.0,
                trap_reward: float = -1.0,
                obstacle_reward: float = -0.1,
                noise_prob: float = 0.2):
        """
        初始化随机网格世界环境
        
        参数:
        height: 网格高度
        width: 网格宽度
        start_pos: 起始位置 (行, 列)
        goal_pos: 目标位置 (行, 列)，如果为None则默认为右下角
        obstacles: 障碍物位置列表 [(行, 列), ...]
        traps: 陷阱位置列表 [(行, 列), ...]
        default_reward: 默认移动奖励
        goal_reward: 到达目标奖励
        trap_reward: 陷阱奖励
        obstacle_reward: 撞到障碍物的奖励
        noise_prob: 动作随机性概率
        """
        # 设置默认的随机性
        super().__init__(
            height=height,
            width=width,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            traps=traps,
            default_reward=default_reward,
            goal_reward=goal_reward,
            trap_reward=trap_reward,
            obstacle_reward=obstacle_reward,
            noise_prob=noise_prob
        )


if __name__ == "__main__":
    # 简单测试
    print("网格世界环境测试")
    
    # 创建一个5x5网格世界
    env = GridEnvironment(
        height=5,
        width=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(1, 1), (2, 1), (3, 1)],
        traps=[(1, 3)],
        default_reward=-0.04,
        noise_prob=0.0  # 确定性环境
    )
    
    # 设置一个简单的策略：总是向右或向下移动
    policy = np.zeros((5, 5), dtype=int)
    for row in range(5):
        for col in range(5):
            if col < 4:
                policy[row, col] = env.RIGHT  # 向右
            else:
                policy[row, col] = env.DOWN  # 向下
    
    env.set_policy(policy)
    
    # 设置一些值函数
    values = np.zeros((5, 5))
    for row in range(5):
        for col in range(5):
            # 简单的启发式：离目标越近价值越高
            distance = abs(row - 4) + abs(col - 4)
            values[row, col] = 1.0 - distance * 0.1
    
    env.set_state_values(values)
    
    # 渲染环境
    env.render()
    
    # 模拟一些步骤
    env.reset()
    print(f"初始位置: {env.current_pos}")
    
    for _ in range(10):
        action = policy[env.current_pos]
        next_state, reward, done = env.step(action)
        print(f"动作: {action}, 新位置: {env.current_pos}, 奖励: {reward}, 终止: {done}")
        
        if done:
            print("到达终止状态")
            break
    
    # 再次渲染环境，显示轨迹
    env.render()
    
    # 测试随机环境
    print("\n随机网格世界环境测试")
    
    env_stochastic = StochasticGridEnvironment(
        height=5,
        width=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(1, 1), (2, 1), (3, 1)],
        traps=[(1, 3)],
        default_reward=-0.04,
        noise_prob=0.2  # 20%的随机性
    )
    
    env_stochastic.set_policy(policy)
    env_stochastic.set_state_values(values)
    
    # 模拟一些步骤
    env_stochastic.reset()
    print(f"初始位置: {env_stochastic.current_pos}")
    
    for _ in range(15):
        action = policy[env_stochastic.current_pos]
        next_state, reward, done = env_stochastic.step(action)
        print(f"动作: {action}, 新位置: {env_stochastic.current_pos}, 奖励: {reward}, 终止: {done}")
        
        if done:
            print("到达终止状态")
            break
    
    # 再次渲染环境，显示轨迹
    env_stochastic.render() 