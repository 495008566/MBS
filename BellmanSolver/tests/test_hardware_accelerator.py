"""
硬件加速器测试模块

该模块测试硬件加速器的功能，包括：
1. 模拟模式下的矩阵操作
2. 硬件模式下的矩阵操作（如果可用）
3. 精确解与近似解对比（验证阻变存储器噪声对收敛的影响）
"""

import os
import sys
import unittest
import numpy as np
import time
from typing import Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # 导入待测试模块
    from BellmanSolver.hardware_interface.hardware_accelerator import HardwareAccelerator
    from BellmanSolver.mbs_core.hardware_interface import MBSHardwareInterface
    from BellmanSolver.mbs_core.bellman_solver import MBSolver
    from BellmanSolver.grid_world.grid_environment import GridEnvironment
    HARDWARE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入硬件模块: {e}")
    HARDWARE_MODULES_AVAILABLE = False

# 全局变量，用于跳过硬件相关测试
HAS_HARDWARE = False


@unittest.skipIf(not HARDWARE_MODULES_AVAILABLE, "硬件模块不可用")
class TestHardwareAccelerator(unittest.TestCase):
    """测试硬件加速器功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        global HAS_HARDWARE
        
        # 创建仿真模式下的加速器
        self.sim_accelerator = HardwareAccelerator(
            use_hardware=False, 
            noise_level=0.01,  # 噪声水平（论文中的δintrinsic）
            enable_noise=False  # 初始禁用噪声（用于精确解）
        )
        
        # 尝试创建硬件模式下的加速器
        try:
            self.hw_accelerator = HardwareAccelerator(
                use_hardware=True,
                noise_level=0.01,  # 噪声水平（论文中的δintrinsic）
                enable_noise=False  # 初始禁用噪声（用于精确解）
            )
            self.has_hardware = self.hw_accelerator.device_manager.is_device_connected()
            # 设置全局变量
            HAS_HARDWARE = self.has_hardware
        except Exception as e:
            print(f"无法初始化硬件加速器: {e}")
            self.has_hardware = False
            HAS_HARDWARE = False
        
        # 创建测试矩阵和向量
        self.test_matrix = np.array([
            [0.1, 0.2, 0.7],  # 状态转移概率（行和为1）
            [0.3, 0.5, 0.2],
            [0.4, 0.1, 0.5]
        ], dtype=np.float32)
        
        self.test_vector = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        
        # 预计算的结果
        self.expected_result = np.dot(self.test_vector, self.test_matrix)
    
    def tearDown(self):
        """测试后的清理工作"""
        if hasattr(self, 'sim_accelerator'):
            self.sim_accelerator.reset()
        
        if hasattr(self, 'hw_accelerator') and hasattr(self, 'has_hardware') and self.has_hardware:
            self.hw_accelerator.reset()
    
    def test_simulation_load_matrix(self):
        """测试仿真模式下加载矩阵"""
        # 加载矩阵
        result = self.sim_accelerator.load_matrix('test_matrix', self.test_matrix)
        
        # 验证加载成功
        self.assertTrue(result)
        self.assertIn('test_matrix', self.sim_accelerator.loaded_data)
        self.assertEqual(self.sim_accelerator.loaded_data['test_matrix']['type'], 'matrix')
    
    def test_simulation_load_vector(self):
        """测试仿真模式下加载向量"""
        # 加载向量
        result = self.sim_accelerator.load_vector('test_vector', self.test_vector)
        
        # 验证加载成功
        self.assertTrue(result)
        self.assertIn('test_vector', self.sim_accelerator.loaded_data)
        self.assertEqual(self.sim_accelerator.loaded_data['test_vector']['type'], 'vector')
    
    def test_simulation_exact_vs_approximate(self):
        """测试精确解与近似解对比（论文Figure 2d,2e）"""
        # 加载数据
        self.sim_accelerator.load_matrix('test_matrix', self.test_matrix)
        self.sim_accelerator.load_vector('test_vector', self.test_vector)
        
        # 执行精确解乘法（禁用噪声）
        self.sim_accelerator.enable_noise_simulation(False)
        exact_result = self.sim_accelerator.vector_matrix_multiply(
            vector_id='test_vector',
            matrix_id='test_matrix'
        )
        
        # 执行近似解乘法（启用噪声）
        self.sim_accelerator.enable_noise_simulation(True)
        approx_result = self.sim_accelerator.vector_matrix_multiply(
            vector_id='test_vector',
            matrix_id='test_matrix'
        )
        
        # 计算结果差异
        diff = np.abs(exact_result - approx_result)
        max_diff = np.max(diff)
        
        # 验证结果在可接受的误差范围内
        self.assertLess(max_diff, 0.1)  # 误差应当小于阈值
        
        # 验证与预期结果的一致性
        np.testing.assert_allclose(exact_result, self.expected_result, rtol=1e-5)
        
        # 打印结果比较
        print("\n精确解 vs 近似解比较:")
        print(f"精确解: {exact_result}")
        print(f"近似解: {approx_result}")
        print(f"最大差异: {max_diff}")
    
    def test_mbs_interface_simulation(self):
        """测试MBS硬件接口在仿真模式下的功能"""
        # 创建硬件接口
        mbs_interface = MBSHardwareInterface(
            state_size=3, 
            use_hardware=False
        )
        
        # 执行乘法
        result = mbs_interface.matrix_vector_multiply(self.test_vector, self.test_matrix)
        
        # 验证结果
        np.testing.assert_allclose(result, self.expected_result, rtol=1e-5)
    
    @unittest.skipIf(not HARDWARE_MODULES_AVAILABLE or not HAS_HARDWARE, "硬件不可用")
    def test_hardware_operations(self):
        """测试真实硬件模式下的操作 (如果硬件可用)"""
        # 加载数据
        self.hw_accelerator.load_matrix('hw_matrix', self.test_matrix)
        self.hw_accelerator.load_vector('hw_vector', self.test_vector)
        
        # 执行乘法
        result = self.hw_accelerator.vector_matrix_multiply(
            vector_id='hw_vector',
            matrix_id='hw_matrix'
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, self.expected_result, rtol=1e-4)
    
    @unittest.skipIf(not HARDWARE_MODULES_AVAILABLE or not HAS_HARDWARE, "硬件不可用")
    def test_mbs_interface_hardware(self):
        """测试MBS硬件接口在真实硬件模式下的功能 (如果硬件可用)"""
        # 创建硬件接口
        mbs_interface = MBSHardwareInterface(state_size=3, use_hardware=True)
        
        # 执行乘法
        result = mbs_interface.matrix_vector_multiply(self.test_vector, self.test_matrix)
        
        # 验证结果
        np.testing.assert_allclose(result, self.expected_result, rtol=1e-4)
    
    def test_noise_impact_on_convergence(self):
        """测试噪声对迭代收敛速度的影响（论文Figure 2d,2e）"""
        # 创建一个简单的5x5网格环境
        env = GridEnvironment(
            height=5,
            width=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[],
            default_reward=-0.04,  # 默认奖励
            goal_reward=1.0,       # 终点奖励
            noise_prob=0.0         # 确定性环境
        )
        
        # 获取状态转移矩阵和奖励向量
        P = env.get_transition_matrix()
        R = env.get_reward_vector()
        exact_solver.preprocess(R, P)
        
        # 记录精确解迭代次数
        exact_mbr_count = 0
        original_mbr_method = exact_solver._mbr_value_iteration
        
        def count_exact_mbr():
            nonlocal exact_mbr_count
            exact_mbr_count += 1
            return original_mbr_method()
        
        exact_solver._mbr_value_iteration = count_exact_mbr
        
        # 计时求解过程
        start_time_exact = time.time()
        exact_V, exact_W = exact_solver.solve(max_iterations=20)
        exact_time = time.time() - start_time_exact
        
        # 重置方法
        exact_solver._mbr_value_iteration = original_mbr_method
        
        # 测试近似解迭代次数（模拟阻变存储器噪声）
        approx_solver 
        # 手动设置启用噪声
        if hasattr(approx_solver, 'hardware_interface') and \
           hasattr(approx_solver.hardware_interface, 'accelerator'):
            approx_solver.hardware_interface.accelerator.enable_noise_simulation(True)
        
        approx_solver.preprocess(R, P)
        
        # 记录近似解迭代次数
        approx_mbr_count = 0
        original_approx_method = approx_solver._mbr_value_iteration
        
        def count_approx_mbr():
            nonlocal approx_mbr_count
            approx_mbr_count += 1
            return original_approx_method()
        
        approx_solver._mbr_value_iteration = count_approx_mbr
        
        # 计时求解过程
        start_time_approx = time.time()
        approx_V, approx_W = approx_solver.solve(max_iterations=20)
        approx_time = time.time() - start_time_approx
        
        # 重置方法
        approx_solver._mbr_value_iteration = original_approx_method
        
        # 验证近似解迭代次数小于精确解
        self.assertLessEqual(approx_mbr_count, exact_mbr_count)
        
        # 验证结果相似性
        value_diff = np.max(np.abs(exact_V - approx_V))
        self.assertLess(value_diff, 0.2)  # 值差异应当小于阈值
        
        # 打印结果
        print("\n噪声对迭代收敛的影响测试:")
        print(f"精确解MBr迭代次数: {exact_mbr_count}")
        print(f"近似解MBr迭代次数: {approx_mbr_count}")
        print(f"迭代减少率: {(exact_mbr_count - approx_mbr_count) / exact_mbr_count:.2%}")
        print(f"精确解求解时间: {exact_time:.6f} 秒")
        print(f"近似解求解时间: {approx_time:.6f} 秒")
        print(f"值函数最大差异: {value_diff:.6f}")
    
    def test_convergence_guarantee(self):
        """测试近似解收敛保证（验证论文中的近似解收敛证明）"""
        # 创建一个简单的值迭代问题
        state_size = 10
        
        # 创建随机转移矩阵（每行和为1）
        P = np.random.rand(state_size, state_size)
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / row_sums
        
        # 创建随机奖励向量
        R = np.random.rand(state_size)
        
        # 设置不同噪声级别
        noise_levels = [0.0, 0.01, 0.05, 0.1]
        
        results = []
        convergence_iterations = []
        
        for noise_level in noise_levels:
            # 创建模拟加速器
            accelerator = HardwareAccelerator(
                use_hardware=False,
                noise_level=noise_level,
                enable_noise=(noise_level > 0)
            )
            
            # 加载矩阵和初始值向量
            accelerator.load_matrix('P', P)
            V = np.zeros(state_size)
            
            # 值迭代
            iterations = 0
            max_iterations = 1000
            converged = False
            previous_V = np.copy(V)
            
            while not converged and iterations < max_iterations:
                iterations += 1
                
                # 加载当前值向量
                accelerator.load_vector('V', V)
                
                # 执行贝尔曼更新 (V = R + γPV)
                gamma = 0.9  # 折扣因子
                PV = accelerator.vector_matrix_multiply('V', 'P')
                V = R + gamma * PV
                
                # 检查收敛
                delta = np.max(np.abs(V - previous_V))
                if delta < 0.01:  # 收敛阈值
                    converged = True
                
                previous_V = np.copy(V)
            
            results.append(V)
            convergence_iterations.append(iterations)
        
        # 验证所有噪声级别都能收敛
        for i, iterations in enumerate(convergence_iterations):
            self.assertLess(iterations, max_iterations, 
                          f"噪声级别 {noise_levels[i]} 未能在最大迭代次数内收敛")
        
        # 验证结果相似性（与无噪声结果比较）
        for i in range(1, len(results)):
            diff = np.max(np.abs(results[0] - results[i]))
            self.assertLess(diff, 0.2, 
                          f"噪声级别 {noise_levels[i]} 的结果与无噪声结果差异过大")
        
        # 打印结果
        print("\n不同噪声级别收敛测试:")
        for i, noise_level in enumerate(noise_levels):
            print(f"噪声级别 {noise_level}: {convergence_iterations[i]} 次迭代")


if __name__ == '__main__':
    unittest.main() 
