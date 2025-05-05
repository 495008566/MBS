"""
MBS硬件接口模块

该模块提供连接MBSolver与硬件加速器的接口，
用于在启用硬件加速时为贝尔曼求解提供硬件支持。
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple

# 导入硬件加速器（使用绝对导入）
from BellmanSolver.hardware_interface.hardware_accelerator import HardwareAccelerator


class MBSHardwareInterface:
    """MBSolver的硬件接口，封装硬件加速器接口"""
    
    def __init__(self, 
                 state_size: int, 
                 use_hardware: bool = True, 
                 config_path: Optional[str] = None):
        """
        初始化MBS硬件接口
        
        参数:
        state_size: 状态空间大小
        use_hardware: 是否使用真实硬件
        config_path: 配置文件路径
        """
        self.state_size = state_size
        self.use_hardware = use_hardware
        
        # 是否已经初始化加速器
        self.is_accelerator_initialized = False
        
        # 硬件加速器实例
        self.accelerator = None
        
        # 矩阵和向量的ID计数
        self.matrix_counter = 0
        self.vector_counter = 0
        
        # 矩阵和向量ID映射
        self.matrix_ids = {}  # 矩阵哈希 -> 矩阵ID
        self.vector_ids = {}  # 向量哈希 -> 向量ID
        
        # 初始化硬件加速器（如果启用）
        if self.use_hardware:
            self._initialize_accelerator(config_path)
    
    def _initialize_accelerator(self, config_path: Optional[str] = None) -> None:
        """
        初始化硬件加速器
        
        参数:
        config_path: 配置文件路径
        """
        try:
            self.accelerator = HardwareAccelerator(use_hardware=self.use_hardware, 
                                                config_path=config_path)
            self.is_accelerator_initialized = True
            print("[MBSHardwareInterface] 硬件加速器初始化成功")
        except Exception as e:
            print(f"[MBSHardwareInterface] 初始化硬件加速器失败: {e}")
            self.is_accelerator_initialized = False
            self.use_hardware = False
    
    def matrix_vector_multiply(self, 
                             vector: np.ndarray, 
                             matrix: np.ndarray) -> np.ndarray:
        """
        执行矩阵-向量乘法，使用硬件加速（如果可用）
        
        参数:
        vector: 输入向量
        matrix: 权重矩阵
        
        返回:
        np.ndarray: 乘法结果
        """
        if not self.use_hardware or not self.is_accelerator_initialized:
            # 如果硬件不可用，直接用numpy计算
            return np.dot(vector, matrix)
        
        try:
            # 为矩阵和向量生成唯一标识
            matrix_hash = hash(matrix.data.tobytes())
            vector_hash = hash(vector.data.tobytes())
            
            # 检查矩阵是否已加载到硬件
            if matrix_hash not in self.matrix_ids:
                matrix_id = f"matrix_{self.matrix_counter}"
                self.matrix_counter += 1
                
                # 加载矩阵
                success = self.accelerator.load_matrix(matrix_id, matrix)
                if not success:
                    print("[MBSHardwareInterface] 加载矩阵到硬件失败，使用CPU计算")
                    return np.dot(vector, matrix)
                
                # 记录矩阵ID
                self.matrix_ids[matrix_hash] = matrix_id
            else:
                matrix_id = self.matrix_ids[matrix_hash]
            
            # 检查向量是否已加载到硬件
            vector_id = f"vector_{self.vector_counter}"
            self.vector_counter += 1
            
            # 加载向量（向量通常每次都不同，不需要缓存）
            success = self.accelerator.load_vector(vector_id, vector)
            if not success:
                print("[MBSHardwareInterface] 加载向量到硬件失败，使用CPU计算")
                return np.dot(vector, matrix)
            
            # 执行乘法
            result_id = f"result_{self.vector_counter}"
            result = self.accelerator.vector_matrix_multiply(
                vector_id=vector_id,
                matrix_id=matrix_id,
                result_id=result_id
            )
            
            # 卸载向量和结果（但保留矩阵）
            self.accelerator.unload_data(vector_id)
            self.accelerator.unload_data(result_id)
            
            if result is None:
                print("[MBSHardwareInterface] 硬件乘法失败，使用CPU计算")
                return np.dot(vector, matrix)
            
            return result
        
        except Exception as e:
            print(f"[MBSHardwareInterface] 硬件乘法错误: {e}，使用CPU计算")
            return np.dot(vector, matrix)
    
    def cleanup(self) -> None:
        """释放硬件资源"""
        if self.is_accelerator_initialized:
            try:
                self.accelerator.reset()
                print("[MBSHardwareInterface] 硬件资源已释放")
            except Exception as e:
                print(f"[MBSHardwareInterface] 释放硬件资源失败: {e}")


# 单例模式，全局硬件接口实例
_hardware_interface_instance = None

def get_hardware_interface(state_size: int, 
                         use_hardware: bool = True, 
                         config_path: Optional[str] = None) -> MBSHardwareInterface:
    """
    获取硬件接口单例实例
    
    参数:
    state_size: 状态空间大小
    use_hardware: 是否使用真实硬件
    config_path: 配置文件路径
    
    返回:
    MBSHardwareInterface: 硬件接口实例
    """
    global _hardware_interface_instance
    
    if _hardware_interface_instance is None:
        _hardware_interface_instance = MBSHardwareInterface(
            state_size=state_size,
            use_hardware=use_hardware,
            config_path=config_path
        )
    
    return _hardware_interface_instance 