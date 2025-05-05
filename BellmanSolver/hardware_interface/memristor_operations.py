"""
阻变存储器操作模块

该模块提供与阻变存储器硬件的直接交互功能，包括数据读取、写入和向量-矩阵乘法操作。
"""

import os
import time
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Any
import ctypes
import logging

# 导入设备管理器
from .device_manager import get_device_manager


class MemristorOperations:
    """提供与阻变存储器交互的操作"""
    
    def __init__(self, use_hardware: bool = True, config_path: Optional[str] = None):
        """
        初始化阻变存储器操作
        
        参数:
        use_hardware: 是否使用真实硬件
        config_path: 配置文件路径
        """
        self.device_manager = get_device_manager(use_hardware, config_path)
        self.logger = logging.getLogger('MemristorOperations')
        self.setup_logger()
        
        # 与设备进行连接
        if not self.device_manager.is_device_connected():
            success = self.device_manager.open_device()
            if not success:
                self.logger.error("无法连接到阻变存储器设备")
        
        # 获取配置
        self.config = self.device_manager.get_config()
        
        # 用于软件模拟时存储数据
        self.simulated_data = {}
    
    def setup_logger(self) -> None:
        """设置日志记录器"""
        # 设置日志级别
        config = self.device_manager.get_config()
        log_level = getattr(logging, config.get('log_level', 'INFO'))
        self.logger.setLevel(log_level)
        
        # 检查是否已有处理器，如果没有则添加
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def read_data(self, address: int, length: int) -> np.ndarray:
        """
        从阻变存储器读取数据
        
        参数:
        address: 起始地址
        length: 读取长度
        
        返回:
        np.ndarray: 读取的数据
        """
        start_time = time.time()
        
        # 检查设备连接
        if not self.device_manager.is_device_connected():
            self.logger.error("设备未连接，无法读取数据")
            return np.zeros(length, dtype=np.float32)
        
        # 使用硬件还是仿真
        if self.device_manager.use_hardware:
            data = self._read_from_hardware(address, length)
        else:
            data = self._read_from_simulation(address, length)
        
        # 更新性能指标
        end_time = time.time()
        self.device_manager.performance_metrics['read_count'] += 1
        self.device_manager.performance_metrics['total_read_time'] += (end_time - start_time)
        
        return data
    
    def _read_from_hardware(self, address: int, length: int) -> np.ndarray:
        """
        从实际硬件读取数据
        
        参数:
        address: 起始地址
        length: 读取长度
        
        返回:
        np.ndarray: 读取的数据
        """
        try:
            # 准备输出数组
            output_data = np.zeros(length, dtype=np.float32)
            output_status = np.zeros(length, dtype=np.int16)
            
            # 获取读取参数
            read_type = self.config['read_type']
            read_voltage = self.config['read_voltage']
            read_pwm_width = self.config['read_pwm_width']
            read_threshold = self.config['read_threshold']
            
            # 调用DLL函数读取数据
            result = self.device_manager.ReadOperateFuc(
                address,                   # 起始位置
                length,                   # 读取长度
                read_voltage,             # 读取电压
                read_pwm_width,           # PWM宽度
                read_threshold,           # 阈值
                read_type,                # 读取类型
                0,                        # 纠错
                0,                        # 数据类型
                output_data,              # 输出数据
                output_status             # 输出状态
            )
            
            if result != 0:
                self.logger.info(f"成功从地址 {address} 读取 {length} 个数据")
                return output_data
            else:
                self.logger.error(f"从地址 {address} 读取数据失败")
                return np.zeros(length, dtype=np.float32)
        
        except Exception as e:
            self.logger.error(f"读取硬件时发生错误: {e}")
            return np.zeros(length, dtype=np.float32)
    
    def _read_from_simulation(self, address: int, length: int) -> np.ndarray:
        """
        从仿真存储中读取数据
        
        参数:
        address: 起始地址
        length: 读取长度
        
        返回:
        np.ndarray: 读取的数据
        """
        result = np.zeros(length, dtype=np.float32)
        
        # 从仿真数据中读取
        for i in range(length):
            key = address + i
            if key in self.simulated_data:
                result[i] = self.simulated_data[key]
        
        self.logger.info(f"从仿真存储中读取 {length} 个数据")
        return result
    
    def write_data(self, address: int, data: np.ndarray) -> bool:
        """
        将数据写入阻变存储器
        
        参数:
        address: 起始地址
        data: 要写入的数据
        
        返回:
        bool: 是否成功写入
        """
        start_time = time.time()
        
        # 检查设备连接
        if not self.device_manager.is_device_connected():
            self.logger.error("设备未连接，无法写入数据")
            return False
        
        # 使用硬件还是仿真
        if self.device_manager.use_hardware:
            success = self._write_to_hardware(address, data)
        else:
            success = self._write_to_simulation(address, data)
        
        # 更新性能指标
        end_time = time.time()
        self.device_manager.performance_metrics['write_count'] += 1
        self.device_manager.performance_metrics['total_write_time'] += (end_time - start_time)
        
        return success
    
    def _write_to_hardware(self, address: int, data: np.ndarray) -> bool:
        """
        将数据写入实际硬件
        
        参数:
        address: 起始地址
        data: 要写入的数据
        
        返回:
        bool: 是否成功写入
        """
        try:
            length = len(data)
            
            # 获取写入参数
            write_type = self.config['write_type']
            pwm_width = self.config['pwm_width']
            valid_width = self.config['valid_width']
            pwm_width_convert = self.config['pwm_width_convert']
            valid_width_convert = self.config['valid_width_convert']
            pwm_count = self.config['pwm_count']
            set_voltage = self.config['set_voltage']
            write_limit = self.config['write_limit']
            
            # 一条一条数据写入
            success = True
            for i in range(length):
                current_data = int(data[i] * pwm_width_convert)
                current_address = address + i
                
                # 调用DLL函数写入
                result = self.device_manager.WritePwmOperateFuc(
                    current_address,      # 地址
                    write_type,           # 写入类型
                    current_data,         # 数据
                    pwm_width,            # PWM宽度
                    valid_width,          # 有效宽度
                    valid_width_convert,  # 有效宽度转换
                    pwm_count,            # PWM计数
                    0,                    # 纠错使能
                    0,                    # 保留参数
                    set_voltage,          # 设置电压
                    write_limit,          # 写入限制
                    False,                # 是否拉伸
                    False                 # 是否抑制
                )
                
                if result == 0:
                    self.logger.error(f"写入地址 {current_address} 失败")
                    success = False
            
            if success:
                self.logger.info(f"成功写入 {length} 个数据到地址 {address}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"写入硬件时发生错误: {e}")
            return False
    
    def _write_to_simulation(self, address: int, data: np.ndarray) -> bool:
        """
        将数据写入仿真存储
        
        参数:
        address: 起始地址
        data: 要写入的数据
        
        返回:
        bool: 是否成功写入
        """
        try:
            length = len(data)
            
            # 写入仿真数据
            for i in range(length):
                self.simulated_data[address + i] = data[i]
            
            self.logger.info(f"成功写入 {length} 个数据到仿真存储，地址 {address}")
            return True
        
        except Exception as e:
            self.logger.error(f"写入仿真存储时发生错误: {e}")
            return False
    
    def vector_matrix_multiply(self, 
                              vector: np.ndarray, 
                              matrix: np.ndarray,
                              matrix_address: Optional[int] = None) -> np.ndarray:
        """
        执行向量-矩阵乘法
        
        参数:
        vector: 输入向量
        matrix: 权重矩阵（仅在仿真模式下使用）
        matrix_address: 硬件中储存矩阵的地址（硬件模式必须提供）
        
        返回:
        np.ndarray: 乘法结果
        """
        start_time = time.time()
        
        # 检查设备连接
        if not self.device_manager.is_device_connected():
            self.logger.error("设备未连接，无法执行向量-矩阵乘法")
            return np.zeros(matrix.shape[1] if matrix is not None else 0, dtype=np.float32)
        
        # 使用硬件还是仿真
        if self.device_manager.use_hardware:
            if matrix_address is None:
                self.logger.error("硬件模式下必须提供矩阵地址")
                return np.zeros(0, dtype=np.float32)
            result = self._vm_multiply_hardware(vector, matrix_address)
        else:
            if matrix is None:
                self.logger.error("仿真模式下必须提供矩阵数据")
                return np.zeros(0, dtype=np.float32)
            result = self._vm_multiply_simulation(vector, matrix)
        
        # 更新性能指标
        end_time = time.time()
        self.device_manager.performance_metrics['vector_matrix_ops'] += 1
        self.device_manager.performance_metrics['total_vm_ops_time'] += (end_time - start_time)
        
        return result
    
    def _vm_multiply_hardware(self, vector: np.ndarray, matrix_address: int) -> np.ndarray:
        """
        使用硬件执行向量-矩阵乘法
        
        参数:
        vector: 输入向量
        matrix_address: 矩阵在硬件中的地址
        
        返回:
        np.ndarray: 乘法结果
        """
        try:
            # 如果我们有RecognitionOperate函数，则使用它
            # 否则，我们需要分步骤进行：读取矩阵，然后在软件中计算
            
            if hasattr(self.device_manager, 'RecognitionOperateFuc'):
                # 准备参数
                vector_length = len(vector)
                threshold = self.config['read_threshold']
                read_type = self.config['read_type']
                
                # RecognitionOperateFuc的参数缺失一些信息，这里是一个简化版
                # 实际代码需要根据确切的函数签名调整
                # 这里假设我们有输出向量的大小
                output_length = 0  # 需要根据矩阵大小确定
                
                # 准备输出数组
                output_data = np.zeros(output_length, dtype=np.float32)
                input_vector = np.array(vector, dtype=np.int16)
                
                # 调用DLL函数执行乘法
                result = self.device_manager.RecognitionOperateFuc(
                    threshold,            # 阈值
                    read_type,            # 读取类型
                    matrix_address,       # 矩阵地址
                    vector_length,        # 向量长度
                    input_vector,         # 输入向量
                    output_data           # 输出数据
                )
                
                if result != 0:
                    self.logger.info("成功执行向量-矩阵乘法")
                    return output_data
                else:
                    self.logger.error("执行向量-矩阵乘法失败")
                    return np.zeros(output_length, dtype=np.float32)
            else:
                # 在软件中模拟执行乘法
                self.logger.warning("硬件接口不支持直接向量-矩阵乘法，将使用软件模拟")
                
                # 假设矩阵的行数等于向量长度
                vector_length = len(vector)
                
                # 读取矩阵（假设矩阵以行优先方式存储）
                # 这里需要知道矩阵的列数，而这通常需要额外信息
                # 假设我们可以从其他地方获取这个信息
                matrix_rows = vector_length
                matrix_cols = 0  # 需要从配置或其他地方获取
                
                matrix_data = self.read_data(matrix_address, matrix_rows * matrix_cols)
                matrix = matrix_data.reshape(matrix_rows, matrix_cols)
                
                # 执行软件乘法
                return np.dot(vector, matrix)
        
        except Exception as e:
            self.logger.error(f"硬件向量-矩阵乘法时发生错误: {e}")
            return np.zeros(0, dtype=np.float32)
    
    def _vm_multiply_simulation(self, vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        在仿真中执行向量-矩阵乘法
        
        参数:
        vector: 输入向量
        matrix: 权重矩阵
        
        返回:
        np.ndarray: 乘法结果
        """
        try:
            # 执行向量-矩阵乘法
            result = np.dot(vector, matrix)
            self.logger.info(f"成功执行向量-矩阵乘法，向量长度 {len(vector)}，矩阵形状 {matrix.shape}")
            return result
        
        except Exception as e:
            self.logger.error(f"仿真向量-矩阵乘法时发生错误: {e}")
            return np.zeros(matrix.shape[1], dtype=np.float32)
    
    def close(self) -> None:
        """关闭设备连接"""
        self.device_manager.close_device()


# 创建单例实例
_memristor_ops_instance = None

def get_memristor_operations(use_hardware: bool = True, config_path: Optional[str] = None) -> MemristorOperations:
    """
    获取阻变存储器操作的全局实例
    
    参数:
    use_hardware: 是否使用真实硬件
    config_path: 配置文件路径
    
    返回:
    MemristorOperations: 阻变存储器操作实例
    """
    global _memristor_ops_instance
    
    if _memristor_ops_instance is None:
        _memristor_ops_instance = MemristorOperations(use_hardware, config_path)
    
    return _memristor_ops_instance


if __name__ == "__main__":
    # 简单测试
    print("阻变存储器操作测试")
    
    # 使用仿真模式
    ops = get_memristor_operations(use_hardware=False)
    
    # 写入一些测试数据
    test_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    success = ops.write_data(100, test_data)
    print(f"写入数据: {'成功' if success else '失败'}")
    
    # 读取数据
    read_data = ops.read_data(100, 5)
    print(f"读取数据: {read_data}")
    
    # 测试向量-矩阵乘法
    vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    matrix = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ], dtype=np.float32)
    
    result = ops.vector_matrix_multiply(vector, matrix)
    print(f"向量-矩阵乘法结果: {result}")
    
    # 关闭连接
    ops.close() 