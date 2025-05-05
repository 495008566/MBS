"""
硬件加速器模块

该模块提供了阻变存储器硬件加速器的抽象接口，
用于在阻变存储器上执行向量-矩阵乘法(VMM)操作，
支持MBS (Memristor-Based Solver)算法。

根据论文"Memristive Bellman solver for decision-making"，
阻变存储器的内在噪声实际上有助于加速贝尔曼方程求解过程。
"""

import os
import time
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Any
import logging

# 导入相关模块
from .device_manager import get_device_manager
from .memristor_operations import get_memristor_operations
from .memory_manager import MemoryManager
from .data_converter import (
    normalize_data, quantize_data, dequantize_data,
    encode_matrix_for_hardware, decode_matrix_from_hardware,
    encode_vector_for_hardware, decode_vector_from_hardware
)


class HardwareAccelerator:
    """
    阻变存储器硬件加速器
    
    为MBS算法提供硬件加速能力，通过阻变存储器执行VMM操作。
    根据论文描述，该加速器利用了阻变存储器的内在噪声特性，
    可以加速贝尔曼方程的收敛过程。
    
    论文的关键创新点之一是利用阻变存储器的内在噪声（δintrinsic~N(0,φ²)）
    加速贝尔曼方程的收敛，从而减少迭代次数。
    """
    
    def __init__(self, 
                use_hardware: bool = True, 
                config_path: Optional[str] = None,
                noise_level: float = 0.01,  # 读取噪声水平（论文中的φ值，标准差）
                enable_noise: bool = True):  # 是否启用噪声（用于近似解）
        """
        初始化硬件加速器
        
        参数:
        use_hardware: 是否使用实际硬件（False则使用软件模拟）
        config_path: 硬件配置文件路径
        noise_level: 读取噪声水平（论文中的φ值，噪声的标准差）
        enable_noise: 是否启用噪声模拟（论文中的近似解方法）
        """
        self.use_hardware = use_hardware
        self.config_path = config_path
        self.noise_level = noise_level  # 读取噪声水平
        self.enable_noise = enable_noise  # 是否启用噪声
        
        # 获取设备管理器和阻变存储器操作接口
        self.device_manager = get_device_manager(use_hardware, config_path)
        self.memristor_ops = get_memristor_operations(use_hardware, config_path)
        
        # 创建内存管理器
        self.memory_manager = MemoryManager()
        
        # 获取配置
        self.config = self.device_manager.get_config()
        
        # 设置日志记录器
        self.logger = logging.getLogger('HardwareAccelerator')
        self.setup_logger()
        
        # 跟踪已加载的数据
        self.loaded_data = {}
        
        # 连接硬件
        if not self.device_manager.is_device_connected():
            success = self.device_manager.open_device()
            if not success:
                self.logger.warning("无法连接到硬件，将使用仿真模式")
                
        # 记录噪声统计信息
        self.noise_stats = {
            "total_operations": 0,
            "noise_enabled_operations": 0,
            "last_noise_magnitude": 0.0,
            "total_noise_magnitude": 0.0
        }
    
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
    
    def load_matrix(self, matrix_id: str, matrix: np.ndarray) -> bool:
        """
        加载矩阵到阻变存储器
        
        根据论文，矩阵对应状态转移概率，映射到阻变存储器的电导状态
        
        参数:
        matrix_id: 矩阵标识符
        matrix: 要加载的矩阵数据
        
        返回:
        bool: 加载是否成功
        """
        if self.use_hardware:
            try:
                # 转换矩阵格式
                hw_matrix = encode_matrix_for_hardware(matrix, conversion_scale=100.0)
                
                # 申请阻变存储器内存
                mem_address = self.memory_manager.register_data(matrix_id, hw_matrix.size * hw_matrix.itemsize)
                
                # 写入数据到阻变存储器
                success = self.memristor_ops.write_data(mem_address, hw_matrix)
                
                if success:
                    # 记录矩阵元数据
                    self.loaded_data[matrix_id] = {
                        'type': 'matrix',
                        'address': mem_address,
                        'shape': matrix.shape,
                        'original': matrix  # 保存原始数据以便比较
                    }
                
                return success
            except Exception as e:
                self.logger.error(f"加载矩阵到硬件失败: {e}")
                return False
        else:
            # 模拟模式，直接存储矩阵
            self.loaded_data[matrix_id] = {
                'type': 'matrix',
                'data': matrix.copy(),
                'shape': matrix.shape
            }
            return True
    
    def load_vector(self, vector_id: str, vector: np.ndarray) -> bool:
        """
        加载向量到阻变存储器
        
        根据论文，向量对应R(St)+γV(St-1)，作为VMM操作的输入
        
        参数:
        vector_id: 向量标识符
        vector: 要加载的向量数据
        
        返回:
        bool: 加载是否成功
        """
        if self.use_hardware:
            try:
                # 转换向量格式
                hw_vector = encode_vector_for_hardware(vector, conversion_scale=100.0)
                
                # 申请阻变存储器内存
                mem_address = self.memory_manager.register_data(vector_id, hw_vector.size * hw_vector.itemsize)
                
                # 写入数据到阻变存储器
                success = self.memristor_ops.write_data(mem_address, hw_vector)
                
                if success:
                    # 记录向量元数据
                    self.loaded_data[vector_id] = {
                        'type': 'vector',
                        'address': mem_address,
                        'shape': vector.shape,
                        'original': vector  # 保存原始数据以便比较
                    }
                
                return success
            except Exception as e:
                self.logger.error(f"加载向量到硬件失败: {e}")
                return False
        else:
            # 模拟模式，直接存储向量
            self.loaded_data[vector_id] = {
                'type': 'vector',
                'data': vector.copy(),
                'shape': vector.shape
            }
            return True
    
    def read_matrix(self, matrix_id: str) -> Optional[np.ndarray]:
        """
        从硬件中读取矩阵
        
        参数:
        matrix_id: 矩阵ID
        
        返回:
        Optional[np.ndarray]: 矩阵数据，如果失败则返回None
        """
        if matrix_id not in self.loaded_data:
            self.logger.error(f"矩阵 {matrix_id} 未加载")
            return None
        
        data_info = self.loaded_data[matrix_id]
        if data_info['type'] != 'matrix':
            self.logger.error(f"数据 {matrix_id} 不是矩阵")
            return None
        
        # 获取地址和形状
        address = data_info['address']
        shape = data_info['shape']
        
        # 计算大小
        size = shape[0] * shape[1]
        
        # 读取数据
        encoded_data = self.memristor_ops.read_data(address, size)
        
        # 解码数据
        matrix = decode_matrix_from_hardware(
            encoded_data, shape, 
            conversion_scale=100.0
        )
        
        self.logger.info(f"矩阵 {matrix_id} 成功从地址 {address} 读取")
        return matrix
    
    def read_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """
        从硬件中读取向量
        
        参数:
        vector_id: 向量ID
        
        返回:
        Optional[np.ndarray]: 向量数据，如果失败则返回None
        """
        if vector_id not in self.loaded_data:
            self.logger.error(f"向量 {vector_id} 未加载")
            return None
        
        data_info = self.loaded_data[vector_id]
        if data_info['type'] != 'vector':
            self.logger.error(f"数据 {vector_id} 不是向量")
            return None
        
        # 获取地址和形状
        address = data_info['address']
        shape = data_info['shape']
        
        # 计算大小
        size = np.prod(shape)
        
        # 读取数据
        encoded_data = self.memristor_ops.read_data(address, size)
        
        # 解码数据
        vector = decode_vector_from_hardware(
            encoded_data, 
            conversion_scale=100.0
        )
        
        # 调整形状
        vector = vector.reshape(shape)
        
        self.logger.info(f"向量 {vector_id} 成功从地址 {address} 读取")
        return vector
    
    def vector_matrix_multiply(self, 
                              vector_id: str, 
                              matrix_id: str,
                              result_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        执行向量-矩阵乘法(MBdot操作)
        
        论文中描述的MBdot操作，利用阻变存储器的模拟计算能力
        执行向量与矩阵的点积。
        
        论文的关键创新点：
        - 利用阻变存储器的内在噪声加速贝尔曼方程求解
        - 噪声分布：δintrinsic~N(0,φ²)，其中φ是噪声标准差
        
        参数:
        vector_id: 输入向量的标识符
        matrix_id: 权重矩阵的标识符
        result_id: 结果向量的标识符（可选）
        
        返回:
        np.ndarray: 乘法结果，如果失败则返回None
        """
        # 更新操作计数
        self.noise_stats["total_operations"] += 1
        
        # 如果启用了噪声，更新噪声统计
        if self.enable_noise:
            self.noise_stats["noise_enabled_operations"] += 1
            
        if self.use_hardware:
            try:
                # 获取数据地址
                vector_meta = self.loaded_data.get(vector_id)
                matrix_meta = self.loaded_data.get(matrix_id)
                
                if not vector_meta or not matrix_meta:
                    self.logger.error(f"错误: 未找到向量({vector_id})或矩阵({matrix_id})")
                    return None
                
                # 执行硬件VMM操作
                hw_result = self.memristor_ops.vector_matrix_multiply(
                    vector_meta['address'],
                    matrix_meta['address'],
                    vector_meta['shape'],
                    matrix_meta['shape']
                )
                
                # 转换回软件格式
                result = self.memristor_ops.to_software_format(hw_result)
                
                # 如果启用噪声，添加阻变存储器内在噪声
                if self.enable_noise:
                    # 根据论文，添加符合高斯分布的读取噪声
                    noise = np.random.normal(0, self.noise_level, result.shape)
                    
                    # 记录噪声统计
                    noise_magnitude = np.mean(np.abs(noise))
                    self.noise_stats["last_noise_magnitude"] = noise_magnitude
                    self.noise_stats["total_noise_magnitude"] += noise_magnitude
                    
                    # 添加噪声到结果
                    result = result + noise
                
                # 如果指定了结果ID，保存结果
                if result_id:
                    self.loaded_data[result_id] = {
                        'type': 'vector',
                        'data': result.copy(),
                        'shape': result.shape,
                        'is_result': True
                    }
                
                return result
            except Exception as e:
                self.logger.error(f"硬件VMM操作失败: {e}")
                # 如果硬件失败，回退到软件模拟
                return self._simulate_vmm(vector_id, matrix_id, result_id)
        else:
            # 模拟模式下的VMM计算
            return self._simulate_vmm(vector_id, matrix_id, result_id)
    
    def _simulate_vmm(self, vector_id: str, matrix_id: str, result_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        软件模拟向量-矩阵乘法
        
        当硬件不可用或出错时，使用此方法进行软件模拟，
        包括模拟阻变存储器的读取噪声。
        
        参数:
        vector_id: 输入向量的标识符
        matrix_id: 权重矩阵的标识符
        result_id: 结果向量的标识符（可选）
        
        返回:
        np.ndarray: 乘法结果，如果失败则返回None
        """
        try:
            vector_data = self.loaded_data.get(vector_id, {}).get('data')
            matrix_data = self.loaded_data.get(matrix_id, {}).get('data')
            
            if vector_data is None or matrix_data is None:
                self.logger.error(f"错误: 未找到向量({vector_id})或矩阵({matrix_id})")
                return None
            
            # 论文中提到的阻变存储器内在噪声特性
            if self.enable_noise:
                # 添加符合高斯分布的读取噪声(论文公式31,32)
                # δintrinsic~N(0,φ²)，其中φ是噪声标准差
                noise = np.random.normal(0, self.noise_level, matrix_data.shape)
                
                # 记录噪声统计
                noise_magnitude = np.mean(np.abs(noise))
                self.noise_stats["last_noise_magnitude"] = noise_magnitude
                self.noise_stats["total_noise_magnitude"] += noise_magnitude
                
                # 添加噪声到矩阵
                matrix_with_noise = matrix_data + noise
                
                # 执行带噪声的VMM (论文中的近似解)
                result = np.dot(vector_data, matrix_with_noise)
                
                self.logger.debug(f"已添加噪声(标准差φ={self.noise_level:.4f})到矩阵进行近似解计算")
            else:
                # 精确解，无噪声
                result = np.dot(vector_data, matrix_data)
                self.logger.debug("使用精确解计算，无噪声")
            
            # 如果指定了结果ID，保存结果
            if result_id:
                self.loaded_data[result_id] = {
                    'type': 'vector',
                    'data': result.copy(),
                    'shape': result.shape,
                    'is_result': True
                }
            
            return result
        except Exception as e:
            self.logger.error(f"模拟VMM操作失败: {e}")
            return None
    
    def unload_data(self, data_id: str) -> bool:
        """
        卸载数据
        
        参数:
        data_id: 要卸载的数据标识符
        
        返回:
        bool: 卸载是否成功
        """
        if data_id in self.loaded_data:
            if self.use_hardware:
                try:
                    # 获取数据元信息
                    data_meta = self.loaded_data[data_id]
                    
                    # 释放阻变存储器内存
                    self.memory_manager.free_memory(data_meta['address'])
                    
                    # 移除元数据
                    del self.loaded_data[data_id]
                    self.logger.info(f"数据 {data_id} 已卸载")
                    return True
                except Exception as e:
                    self.logger.error(f"卸载硬件数据失败: {e}")
                    return False
            else:
                # 模拟模式直接删除数据
                del self.loaded_data[data_id]
                self.logger.info(f"数据 {data_id} 已卸载")
                return True
        else:
            self.logger.warning(f"警告: 试图卸载不存在的数据({data_id})")
            return False
    
    def get_loaded_data_info(self) -> Dict[str, Any]:
        """
        获取已加载数据的信息
        
        返回:
        Dict: 数据信息
        """
        return {
            'loaded_data': self.loaded_data,
            'memory_utilization': self.memory_manager.get_memory_utilization(),
            'performance_metrics': self.device_manager.get_performance_metrics(),
            'noise_stats': self.noise_stats
        }
    
    def defragment_memory(self) -> bool:
        """
        整理内存碎片
        
        返回:
        bool: 是否成功整理
        """
        # 执行碎片整理
        data_moves = self.memory_manager.defragment()
        
        if not data_moves:
            self.logger.info("无需整理内存碎片")
            return True
        
        # 移动数据
        for data_id, move_info in data_moves.items():
            old_address = move_info['old_address']
            new_address = move_info['new_address']
            size = move_info['size']
            
            # 读取数据
            data = self.memristor_ops.read_data(old_address, size // 4)  # 假设每个元素为4字节
            
            # 写入新位置
            success = self.memristor_ops.write_data(new_address, data)
            
            if not success:
                self.logger.error(f"移动数据 {data_id} 失败")
                return False
            
            # 更新加载的数据信息
            if data_id in self.loaded_data:
                self.loaded_data[data_id]['address'] = new_address
        
        self.logger.info(f"成功移动 {len(data_moves)} 个数据块")
        return True
    
    def reset(self) -> bool:
        """
        重置硬件加速器
        
        返回:
        bool: 重置是否成功
        """
        if self.use_hardware:
            try:
                # 重置所有硬件组件
                self.device_manager.close_device()
                self.memory_manager.reset()
                self.loaded_data = {}
                
                # 重置噪声统计
                self.noise_stats = {
                    "total_operations": 0,
                    "noise_enabled_operations": 0,
                    "last_noise_magnitude": 0.0,
                    "total_noise_magnitude": 0.0
                }
                
                self.logger.info("硬件加速器已重置")
                return True
            except Exception as e:
                self.logger.error(f"重置硬件失败: {e}")
                return False
        else:
            # 模拟模式直接清空数据
            self.loaded_data = {}
            
            # 重置噪声统计
            self.noise_stats = {
                "total_operations": 0,
                "noise_enabled_operations": 0,
                "last_noise_magnitude": 0.0,
                "total_noise_magnitude": 0.0
            }
            
            self.logger.info("硬件加速器已重置")
            return True
    
    def adjust_noise_level(self, level: float) -> None:
        """
        调整噪声水平
        
        根据论文，噪声水平φ控制了阻变存储器读取噪声的标准差，
        噪声分布为δintrinsic~N(0,φ²)
        
        参数:
        level: 新的噪声标准差（φ值）
        """
        if level < 0:
            self.logger.warning(f"噪声水平必须非负，使用绝对值 |{level}|")
            level = abs(level)
            
        self.noise_level = level
        self.logger.info(f"噪声水平已调整为 φ={level:.4f}，噪声分布为 N(0,{level*level:.6f})")
    
    def enable_noise_simulation(self, enable: bool) -> None:
        """
        启用或禁用噪声模拟
        
        论文中提到了利用阻变存储器的内在噪声加速计算过程，
        此功能允许切换精确解和近似解。
        
        参数:
        enable: 是否启用噪声模拟
        """
        self.enable_noise = enable
        if enable:
            self.logger.info(f"已启用噪声模拟(φ={self.noise_level:.4f})，使用近似解")
        else:
            self.logger.info("已禁用噪声模拟，使用精确解")
    
    def get_noise_statistics(self) -> Dict[str, Any]:
        """
        获取噪声统计信息
        
        返回:
        Dict: 噪声统计信息
        """
        stats = self.noise_stats.copy()
        
        # 计算平均噪声大小
        if stats["noise_enabled_operations"] > 0:
            stats["average_noise_magnitude"] = stats["total_noise_magnitude"] / stats["noise_enabled_operations"]
        else:
            stats["average_noise_magnitude"] = 0.0
            
        # 添加当前噪声配置
        stats["current_noise_level"] = self.noise_level
        stats["noise_enabled"] = self.enable_noise
        
        return stats


if __name__ == "__main__":
    # 测试硬件加速器
    print("测试阻变存储器硬件加速器...")
    
    # 创建加速器实例
    accelerator = HardwareAccelerator(use_hardware=False, enable_noise=True)
    
    # 创建测试矩阵和向量
    test_matrix = np.array([
        [0.1, 0.2, 0.7],
        [0.3, 0.5, 0.2],
        [0.4, 0.1, 0.5]
    ])
    
    test_vector = np.array([0.5, 0.3, 0.2])
    
    # 加载数据
    accelerator.load_matrix('test_matrix', test_matrix)
    accelerator.load_vector('test_vector', test_vector)
    
    # 执行精确解VMM
    accelerator.enable_noise_simulation(False)
    exact_result = accelerator.vector_matrix_multiply('test_vector', 'test_matrix')
    print("精确解结果:")
    print(exact_result)
    
    # 执行近似解VMM（有噪声）
    accelerator.enable_noise_simulation(True)
    approx_result = accelerator.vector_matrix_multiply('test_vector', 'test_matrix')
    print("近似解结果:")
    print(approx_result)
    
    # 比较结果差异
    diff = np.abs(exact_result - approx_result)
    print("差异:")
    print(diff)
    
    # 重置加速器
    accelerator.reset() 