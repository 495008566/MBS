"""
设备管理模块

该模块负责管理与阻变存储器硬件的连接，提供设备初始化、资源管理和状态监控功能。
"""

import os
import time
from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np
import ctypes
from ctypes import cdll
import sys
import logging


class DeviceManager:
    """管理与阻变存储器设备的交互"""
    
    def __init__(self, use_hardware: bool = True, config_path: Optional[str] = None):
        """
        初始化设备管理器
        
        参数:
        use_hardware: 是否使用真实硬件（否则使用仿真模式）
        config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.use_hardware = use_hardware
        self.device_handle = None
        self.is_connected = False
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # 记录设备性能指标
        self.performance_metrics = {
            'read_count': 0,
            'write_count': 0,
            'vector_matrix_ops': 0,
            'total_read_time': 0.0,
            'total_write_time': 0.0,
            'total_vm_ops_time': 0.0
        }
        
        if self.use_hardware:
            self._load_hardware_libraries()
        else:
            self.logger.info("使用仿真模式，不连接实际硬件")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置文件
        
        参数:
        config_path: 配置文件路径
        
        返回:
        Dict: 配置字典
        """
        # 默认配置
        default_config = {
            'device_type': 2,  # 默认设备类型
            'read_voltage': 0.25,  # 读取电压
            'read_pwm_width': 2000,  # 读取PWM宽度
            'set_voltage': 2.0,  # 设置电压
            'write_type': 1,  # 写入类型
            'read_threshold': -900,  # 读取阈值
            'read_type': 1,  # 读取类型
            'pwm_width': 1000,  # PWM宽度
            'valid_width': 500,  # 有效宽度
            'pwm_width_convert': 1,  # PWM宽度转换
            'valid_width_convert': 1,  # 有效宽度转换
            'pwm_count': 1,  # PWM计数
            'max_sample': 800,  # 最大样本数
            'write_limit': 1,  # 写入限制
            'log_level': 'INFO'  # 日志级别
        }
        
        # 如果提供了配置文件，则从文件加载配置
        if config_path and os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        
        return default_config
    
    def _setup_logger(self) -> logging.Logger:
        """
        设置日志记录器
        
        返回:
        logging.Logger: 日志记录器
        """
        logger = logging.getLogger('DeviceManager')
        
        # 设置日志级别
        log_level = getattr(logging, self.config['log_level'])
        logger.setLevel(log_level)
        
        # 创建控制台处理器
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(handler)
        
        return logger
    
    def _load_hardware_libraries(self) -> None:
        """加载硬件交互所需的库文件"""
        try:
            # 确定库文件路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            library_path = os.path.join(current_dir, '..', '..', 'DataReadWriteApp.dll')
            absolute_path = os.path.abspath(library_path)
            
            # 如果当前目录不存在，尝试从当前工作目录加载
            if not os.path.exists(absolute_path):
                absolute_path = 'DataReadWriteApp.dll'
            
            self.logger.info(f"加载库文件: {absolute_path}")
            
            # 加载库文件
            try:
                self.rw_dll = cdll.LoadLibrary(absolute_path)
                self.logger.info("库文件加载成功")
            except Exception as e:
                self.logger.error(f"加载库文件失败: {e}")
                self.use_hardware = False
                return
            
            # 定义函数签名
            self._define_function_signatures()
            
        except Exception as e:
            self.logger.error(f"初始化硬件库失败: {e}")
            self.use_hardware = False
    
    def _define_function_signatures(self) -> None:
        """定义库函数的参数和返回类型"""
        try:
            # 初始化设备函数
            self.InitializeDevicesFuc = self.rw_dll.InitializeDevices
            self.InitializeDevicesFuc.argtypes = [ctypes.c_int]
            self.InitializeDevicesFuc.restype = ctypes.c_ulong
            
            # 读取操作函数
            self.ReadOperateFuc = self.rw_dll.ReadOperate
            self.ReadOperateFuc.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags="C_CONTIGUOUS")
            ]
            self.ReadOperateFuc.restype = ctypes.c_ulong
            
            # 写入PWM操作函数
            self.WritePwmOperateFuc = self.rw_dll.WritePwmOperate
            self.WritePwmOperateFuc.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_bool,
                ctypes.c_bool
            ]
            self.WritePwmOperateFuc.restype = ctypes.c_ulong
            
            # 识别操作函数
            self.RecognitionOperateFuc = self.rw_dll.RecognitionOperate
            self.RecognitionOperateFuc.argtypes = [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
            ]
            self.RecognitionOperateFuc.restype = ctypes.c_ulong
            
            # 关闭卡片函数
            self.CloseCardFuc = self.rw_dll.CloseCard
            self.CloseCardFuc.restype = ctypes.c_ulong
            
            self.logger.info("函数签名定义成功")
            
        except Exception as e:
            self.logger.error(f"定义函数签名失败: {e}")
            self.use_hardware = False
    
    def open_device(self) -> bool:
        """
        打开设备连接
        
        返回:
        bool: 是否成功打开设备
        """
        if not self.use_hardware:
            self.logger.info("仿真模式，不需要打开设备")
            self.is_connected = True
            return True
        
        try:
            device_type = self.config['device_type']
            result = self.InitializeDevicesFuc(device_type)
            
            if result != 0:
                self.logger.info("设备打开成功")
                self.is_connected = True
                return True
            else:
                self.logger.error("设备打开失败")
                return False
        
        except Exception as e:
            self.logger.error(f"打开设备时发生错误: {e}")
            return False
    
    def close_device(self) -> bool:
        """
        关闭设备连接
        
        返回:
        bool: 是否成功关闭设备
        """
        if not self.use_hardware or not self.is_connected:
            self.logger.info("无需关闭设备或设备未连接")
            self.is_connected = False
            return True
        
        try:
            result = self.CloseCardFuc()
            
            if result != 0:
                self.logger.info("设备关闭成功")
                self.is_connected = False
                return True
            else:
                self.logger.error("设备关闭失败")
                return False
        
        except Exception as e:
            self.logger.error(f"关闭设备时发生错误: {e}")
            return False
    
    def is_device_connected(self) -> bool:
        """
        检查设备是否已连接
        
        返回:
        bool: 设备是否已连接
        """
        return self.is_connected
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        返回:
        Dict: 包含性能指标的字典
        """
        metrics = self.performance_metrics.copy()
        
        # 计算平均时间
        if metrics['read_count'] > 0:
            metrics['avg_read_time'] = metrics['total_read_time'] / metrics['read_count']
        
        if metrics['write_count'] > 0:
            metrics['avg_write_time'] = metrics['total_write_time'] / metrics['write_count']
        
        if metrics['vector_matrix_ops'] > 0:
            metrics['avg_vm_ops_time'] = metrics['total_vm_ops_time'] / metrics['vector_matrix_ops']
        
        return metrics
    
    def reset_performance_metrics(self) -> None:
        """重置性能指标"""
        for key in self.performance_metrics:
            self.performance_metrics[key] = 0
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        返回:
        Dict: 配置字典
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置
        
        参数:
        new_config: 新的配置字典
        """
        self.config.update(new_config)
        
        # 更新日志级别
        if 'log_level' in new_config:
            log_level = getattr(logging, self.config['log_level'])
            self.logger.setLevel(log_level)
            for handler in self.logger.handlers:
                handler.setLevel(log_level)


# 创建单例实例
_device_manager_instance = None

def get_device_manager(use_hardware: bool = True, config_path: Optional[str] = None) -> DeviceManager:
    """
    获取设备管理器的全局实例
    
    参数:
    use_hardware: 是否使用真实硬件
    config_path: 配置文件路径
    
    返回:
    DeviceManager: 设备管理器实例
    """
    global _device_manager_instance
    
    if _device_manager_instance is None:
        _device_manager_instance = DeviceManager(use_hardware, config_path)
    
    return _device_manager_instance


if __name__ == "__main__":
    # 简单测试
    print("设备管理器测试")
    
    # 使用仿真模式
    manager = get_device_manager(use_hardware=False)
    print(f"设备连接状态: {manager.is_device_connected()}")
    
    # 打开设备
    success = manager.open_device()
    print(f"打开设备: {'成功' if success else '失败'}")
    print(f"设备连接状态: {manager.is_device_connected()}")
    
    # 获取配置
    config = manager.get_config()
    print("\n当前配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 更新配置
    manager.update_config({'read_voltage': 0.3, 'log_level': 'DEBUG'})
    print("\n更新后的配置:")
    config = manager.get_config()
    for key, value in config.items():
        if key in ['read_voltage', 'log_level']:
            print(f"  {key}: {value}")
    
    # 关闭设备
    success = manager.close_device()
    print(f"关闭设备: {'成功' if success else '失败'}")
    print(f"设备连接状态: {manager.is_device_connected()}") 