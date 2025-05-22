"""
贝尔曼解算器核心实现(MBS - Memristor-Based Solver)

此模块实现了基于论文"Memristive Bellman solver for decision-making"中描述的MBS算法，
用于解决强化学习中的贝尔曼方程，可选择性地利用阻变存储器硬件加速计算过程。
通过引入时间维度并将迭代求解过程转换为循环点积操作，实现了与MCIM技术的兼容。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import random
import time
import sys

# 添加项目根目录到路径
sys.path.insert(0, '.')

# 导入硬件接口(使用绝对导入)
try:
    from BellmanSolver.mbs_core.hardware_interface import get_hardware_interface
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    try:
        from .hardware_interface import get_hardware_interface
    except ImportError:
        print("警告: 无法导入硬件接口模块，硬件加速功能将不可用")
        # 创建一个空接口以避免后续代码错误
        def get_hardware_interface(*args, **kwargs):
            print("硬件接口不可用")
            return None

class Logger:
    """简单的日志记录器，后续会被utils.logger替代"""
    def __init__(self):
        pass
    
    def info(self, message):
        print(f"[INFO] {message}")
        
    def debug(self, message):
        print(f"[DEBUG] {message}")
        
    def warning(self, message):
        print(f"[WARNING] {message}")
        
    def error(self, message):
        print(f"[ERROR] {message}")


class MBSolver:
    """
    基于论文中的MBS (Memristor-Based Solver)算法实现
    
    MBS通过引入时间维度并将传统贝尔曼方程转换为循环点积形式，
    使其能够在阻变存储器CIM(Computing-in-Memory)架构上高效实现。
    
    此实现包含两个核心操作：
    1. MBdot - 在阻变存储器上执行向量-矩阵乘法
    2. MBr - 递归执行MBdot操作直到值函数收敛
    
    论文中的创新点：
    - 引入时间维度，将贝尔曼方程转换为循环点积形式: V(St)=[R(St)+γV(St-1)]·∑P(St-1|St)
    - 利用阻变存储器内在噪声加速收敛: δintrinsic~N(0,φ²)
    """
    def __init__(self, 
                 state_size: int, 
                 gamma: float,      # 折扣因子（论文中的γ）- 控制未来奖励的权重
                 epsilon: float,     # MBr收敛阈值（论文中的ε）- 控制值函数迭代的收敛条件
                 tau: float ,         # 权重矩阵收敛阈值（论文中的τ）- 控制策略迭代的收敛条件
                 initial_exploration: float ,  # 初始探索率 - ε-贪婪策略的初始探索概率
                 decay: float ,      # 探索率衰减因子 - 控制探索率随时间衰减的速度
                 min_exploration: float ,    # 最小探索率 - 探索率的下限，确保始终保持一定的探索
                 use_hardware: bool ,       # 是否使用硬件加速
                 noise_level: float ,        # 噪声级别（论文中的φ）- 控制阻变存储器读取噪声的标准差
                 enable_noise: bool = True,        # 是否启用噪声模拟 - 用于切换精确解和近似解
                 config_path: Optional[str] ,  # 硬件配置文件路径
                 logger: Optional[Logger] ):
        """
        初始化MBS求解器
        
        参数:
        state_size: 状态空间大小
        gamma: 折扣因子（论文中的γ）- 控制未来奖励的权重
        epsilon: MBr收敛阈值（论文中的ε）- 控制值函数迭代的收敛条件
        tau: 权重矩阵收敛阈值（论文中的τ）- 控制策略迭代的收敛条件
        initial_exploration: 初始探索率（ε-贪婪策略参数）
        decay: 探索率衰减因子
        min_exploration: 最小探索率
        use_hardware: 是否使用硬件加速
        noise_level: 噪声级别（论文中的φ）- 控制阻变存储器读取噪声的标准差
        enable_noise: 是否启用噪声模拟 - 用于切换精确解和近似解
        config_path: 硬件配置文件路径
        logger: 日志记录器
        """
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.epsilon_exploration = initial_exploration
        self.decay = decay
        self.min_exploration = min_exploration
        self.use_hardware = use_hardware
        self.noise_level = noise_level
        self.enable_noise = enable_noise
        self.logger = logger or Logger()
        
        # 将在预处理阶段初始化
        self.V = None  # 值向量
        self.V_goal = None  # 目标状态的值
        self.W = None  # 权重矩阵（论文中的状态转移概率）
        self.R = None  # 奖励向量
        self.P = None  # 状态转移概率矩阵
        self.time_stamp = None  # 每个状态到目标的距离（论文中的时间维度t）
        self.distance_layers = None  # 按距离分组的状态
        self.d_max = 0  # 最大距离（时间维度上的最大步长）
        
        # 跟踪MBr和MBdot操作的计数器
        self.mbr_iterations = 0
        self.mbdot_operations = 0
        
        # 初始化硬件接口（如果启用硬件加速）
        if self.use_hardware:
            try:
                self.hardware_interface = get_hardware_interface(
                    state_size=state_size,
                    use_hardware=use_hardware,
                    config_path=config_path
                )
                
                # 如果有硬件接口，设置噪声参数
                if hasattr(self.hardware_interface, 'accelerator'):
                    self.hardware_interface.accelerator.adjust_noise_level(self.noise_level)
                    self.hardware_interface.accelerator.enable_noise_simulation(self.enable_noise)
                    
                self.logger.info("硬件接口初始化成功")
            except Exception as e:
                self.logger.error(f"初始化硬件接口失败: {e}")
                self.use_hardware = False
    
    def preprocess(self, R: np.ndarray, P: np.ndarray) -> None:
        """
        预处理阶段: 计算距离并初始化数据结构
        
        参数:
        R: 奖励向量 [S]
        P: 状态转移概率矩阵 [S x S]
        """
        self.R = R
        self.P = P
        
        # 1. 计算每个状态到目标的距离（对应论文中的时间维度t）
        self.time_stamp = self._compute_distance_to_goal()
        
        # 2. 按距离分组状态（论文中的时间维度分组）
        self.distance_layers = self._group_states_by_distance()
        self.d_max = max(self.time_stamp)
        
        # 3. 初始化值向量和权重矩阵
        self.V = np.zeros(self.state_size)
        self.V_goal = 1.0  # 目标状态赋予高值
        
        # 初始化权重矩阵 - 在转移可能的地方设为均匀值
        # 论文中将状态转移概率映射到权重矩阵
        self.W = np.zeros_like(self.P)
        mask = (self.P > 0)
        row_sums = mask.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        self.W = mask / row_sums
        
        self.logger.info(f"预处理完成: 计算了状态距离(最大距离: {self.d_max})并初始化了权重矩阵")
    
    def _compute_distance_to_goal(self) -> np.ndarray:
        """
        计算每个状态到目标状态的最短距离（实现论文中的时间维度t）
        
        返回:
        np.ndarray: 每个状态到目标的距离（时间戳）
        """
        # 找到目标状态(奖励最高的状态)
        goal_state = np.argmax(self.R)
        
        # 使用BFS计算距离
        distances = np.full(self.state_size, -1, dtype=int)
        distances[goal_state] = 0
        queue = [goal_state]
        
        while queue:
            current = queue.pop(0)
            # 找到所有能够到达当前状态的前驱状态
            predecessors = np.where(self.P[:, current] > 0)[0]
            for pred in predecessors:
                if distances[pred] == -1:  # 尚未访问
                    distances[pred] = distances[current] + 1
                    queue.append(pred)
        
        # 处理不可达状态
        distances[distances == -1] = self.state_size
        
        return distances
    
    def _group_states_by_distance(self) -> List[List[int]]:
        """
        按到目标的距离对状态分组（对应论文中的时间维度分组）
        
        返回:
        List[List[int]]: 按距离分组的状态列表
        """
        max_dist = max(self.time_stamp)
        distance_layers = [[] for _ in range(max_dist + 1)]
        
        for state, dist in enumerate(self.time_stamp):
            distance_layers[dist].append(state)
            
        return distance_layers
    
    def solve(self, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        执行MBS算法求解贝尔曼方程
        
        论文中描述的完整MBS算法流程，包括：
        1. 迭代MBr操作直到值函数收敛
        2. 使用ε-贪婪策略更新权重矩阵
        3. 重复以上过程直到权重矩阵收敛
        
        参数:
        max_iterations: 最大迭代次数
        
        返回:
        Tuple[np.ndarray, np.ndarray, Dict]: (值向量V, 权重矩阵W, 统计信息)
        """
        # 重置统计计数器
        self.mbr_iterations = 0
        self.mbdot_operations = 0
        
        start_time = time.time()
        converged = False
        iteration = 0
        
        while not converged and iteration < max_iterations:
            iteration += 1
            self.logger.info(f"开始第 {iteration} 次策略迭代")
            
            # MBr操作 - 迭代值函数计算直到收敛（论文中的MBr operator）
            V = self._mbr_value_iteration()
            
            # 使用ε-贪婪策略更新权重矩阵（论文Figure 2b中的决策优化过程）
            W_new = self._update_weights_epsilon_greedy()
            
            # 检查收敛性（论文中的τ阈值判断）
            if np.max(np.abs(W_new - self.W)) < self.tau:
                converged = True
                self.logger.info("权重矩阵已收敛，达到最优决策")
            else:
                # 更新权重矩阵
                self.W = W_new
                # 衰减探索率
                self.epsilon_exploration = max(self.epsilon_exploration * self.decay, 
                                             self.min_exploration)
                self.logger.info(f"更新探索率为 {self.epsilon_exploration:.4f}")
        
        end_time = time.time()
        solution_time = end_time - start_time
        
        if not converged:
            self.logger.warning(f"未在 {max_iterations} 次迭代内收敛")
        else:
            self.logger.info(f"在 {iteration} 次迭代后收敛到最优决策")
        
        # 返回统计信息
        stats = {
            "mbr_iterations": self.mbr_iterations,
            "mbdot_operations": self.mbdot_operations,
            "policy_iterations": iteration,
            "solution_time": solution_time,
            "converged": converged,
            "noise_enabled": self.enable_noise,
            "noise_level": self.noise_level,
            "hardware_used": self.use_hardware
        }
            
        return self.V, self.W, stats
    
    def _mbr_value_iteration(self) -> np.ndarray:
        """
        MBr: 循环值函数计算直到收敛
        
        论文中描述的MBr操作，通过重复MBdot操作直到值函数收敛
        
        返回:
        np.ndarray: 收敛后的值向量
        """
        converged_mbr = False
        V_prev = np.zeros_like(self.V)
        iterations = 0
        max_iterations = 1000  # 防止无限循环
        
        while not converged_mbr and iterations < max_iterations:
            iterations += 1
            self.mbr_iterations += 1
            
            # 从奖励向量开始（论文Figure 2a中的初始输入）
            input_vec = np.copy(self.R)
            
            # 通过时间维度传播（论文中的时间维度t）
            # 这是论文的关键创新点：将贝尔曼方程转换为循环点积形式
            for t in range(self.d_max + 1):
                # 记录MBdot操作次数
                self.mbdot_operations += 1
                
                # 执行MBdot操作（论文中的MBdot operator）
                if self.use_hardware:
                    # 使用硬件接口执行乘法
                    output_vec = self._mb_dot(input_vec, self.W)
                else:
                    # 软件模拟MBdot操作，如果启用噪声则添加高斯噪声
                    if self.enable_noise:
                        # 根据论文，添加符合高斯分布的读取噪声δintrinsic~N(0,φ²)
                        noise = np.random.normal(0, self.noise_level, self.W.shape)
                        W_with_noise = self.W + noise
                        output_vec = np.dot(input_vec, W_with_noise)
                    else:
                        # 精确解，无噪声
                        output_vec = np.dot(input_vec, self.W)
                
                # 更新输入向量: R(S) + γV(S)（论文公式9中的计算）
                input_vec = self.R + self.gamma * output_vec
            
            # 当前MBr结果
            self.V = output_vec
            
            # 检查收敛性（论文中的ε阈值判断）
            if np.max(np.abs(self.V - V_prev)) < self.epsilon:
                converged_mbr = True
                self.logger.debug(f"MBr值迭代在 {iterations} 次迭代后收敛")
            else:
                V_prev = np.copy(self.V)
        
        if not converged_mbr:
            self.logger.warning(f"MBr值迭代未在 {max_iterations} 次迭代内收敛")
            
        return self.V
    
    def _mb_dot(self, vec_in: np.ndarray, W_matrix: np.ndarray) -> np.ndarray:
        """
        MBdot: 在阻变存储器上执行向量-矩阵乘法
        
        论文中描述的MBdot操作，利用阻变存储器计算V·P
        
        参数:
        vec_in: 输入向量（论文中的R(St)+γV(St-1)）
        W_matrix: 权重矩阵（论文中的状态转移概率P(St-1|St)）
        
        返回:
        np.ndarray: 乘法结果
        """
        if self.use_hardware:
            try:
                # 使用硬件接口执行乘法（阻变存储器的点积操作）
                # 论文中描述的将点积运算映射到阻变存储器阵列上
                return self.hardware_interface.matrix_vector_multiply(vec_in, W_matrix)
            except Exception as e:
                self.logger.error(f"硬件乘法失败: {e}，使用CPU计算")
                if self.enable_noise:
                    # 添加噪声模拟阻变存储器特性
                    noise = np.random.normal(0, self.noise_level, W_matrix.shape)
                    W_with_noise = W_matrix + noise
                    return np.dot(vec_in, W_with_noise)
                else:
                    return np.dot(vec_in, W_matrix)
        else:
            # 软件模拟模式
            if self.enable_noise:
                # 添加符合高斯分布的读取噪声，模拟阻变存储器特性
                noise = np.random.normal(0, self.noise_level, W_matrix.shape)
                W_with_noise = W_matrix + noise
                return np.dot(vec_in, W_with_noise)
            else:
                # 精确解，无噪声
                return np.dot(vec_in, W_matrix)
    
    def _update_weights_epsilon_greedy(self) -> np.ndarray:
        """
        使用ε-贪婪策略更新权重矩阵
        
        论文中的决策优化过程（Figure 2b和论文行178-182）
        
        返回:
        np.ndarray: 新的权重矩阵
        """
        W_new = np.zeros_like(self.W)
        
        # 对每个状态更新权重
        for st in range(self.state_size):
            # 收集所有可能的前驱状态
            pred_states = np.where(self.P[:, st] > 0)[0]
            
            if len(pred_states) > 0:
                # ε-贪婪选择（论文中提到的ε-greedy规则）
                if np.random.random() < self.epsilon_exploration:
                    # 探索: 随机选择一个前驱
                    s_best = np.random.choice(pred_states)
                else:
                    # 利用: 选择值最高的前驱
                    s_best = pred_states[np.argmax(self.V[pred_states])]
                
                # 设置转移权重
                W_new[s_best, st] = 1
        
        # 确保每行和为1(有效概率)
        row_sums = W_new.sum(axis=1, keepdims=True)
        valid_rows = row_sums.flatten() > 0
        W_new[valid_rows] = W_new[valid_rows] / row_sums[valid_rows]
        
        return W_new
    
    def set_noise_parameters(self, noise_level: float = None, enable_noise: bool = None) -> None:
        """
        设置噪声参数
        
        参数:
        noise_level: 噪声级别（论文中的φ值）
        enable_noise: 是否启用噪声（切换精确解和近似解）
        """
        if noise_level is not None:
            self.noise_level = noise_level
            if self.use_hardware and hasattr(self.hardware_interface, 'accelerator'):
                self.hardware_interface.accelerator.adjust_noise_level(noise_level)
        
        if enable_noise is not None:
            self.enable_noise = enable_noise
            if self.use_hardware and hasattr(self.hardware_interface, 'accelerator'):
                self.hardware_interface.accelerator.enable_noise_simulation(enable_noise)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        获取求解统计信息
        
        返回:
        Dict: 统计信息
        """
        return {
            "mbr_iterations": self.mbr_iterations,
            "mbdot_operations": self.mbdot_operations,
            "noise_enabled": self.enable_noise,
            "noise_level": self.noise_level,
            "hardware_used": self.use_hardware
        }
    
    def cleanup(self) -> None:
        """清理资源，释放硬件"""
        if self.use_hardware:
            try:
                self.hardware_interface.cleanup()
                self.logger.info("硬件资源已释放")
            except Exception as e:
                self.logger.error(f"释放硬件资源失败: {e}")

