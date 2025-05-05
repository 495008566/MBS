"""
数据转换模块

该模块提供各种数据格式转换功能，用于在软件数据表示和硬件数据表示之间进行转换。
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any


def normalize_data(data: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """
    将数据归一化到指定范围
    
    参数:
    data: 输入数据
    min_val: 目标最小值
    max_val: 目标最大值
    
    返回:
    np.ndarray: 归一化后的数据
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    # 避免除零
    if data_min == data_max:
        return np.ones_like(data) * min_val
    
    # 归一化
    normalized = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
    
    return normalized


def quantize_data(data: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    将数据量化为指定位数
    
    参数:
    data: 输入数据
    bits: 量化位数
    
    返回:
    np.ndarray: 量化后的数据
    """
    # 计算量化级别数
    levels = 2 ** bits
    
    # 找出数据范围
    data_min = np.min(data)
    data_max = np.max(data)
    
    # 避免除零
    if data_min == data_max:
        return np.zeros_like(data, dtype=np.int32)
    
    # 量化
    scaled = (data - data_min) / (data_max - data_min) * (levels - 1)
    quantized = np.round(scaled).astype(np.int32)
    
    return quantized


def dequantize_data(quantized: np.ndarray, original_min: float, original_max: float, bits: int = 8) -> np.ndarray:
    """
    将量化数据转换回原始数据范围
    
    参数:
    quantized: 量化数据
    original_min: 原始数据最小值
    original_max: 原始数据最大值
    bits: 量化位数
    
    返回:
    np.ndarray: 反量化后的数据
    """
    # 计算量化级别数
    levels = 2 ** bits
    
    # 反量化
    dequantized = quantized / (levels - 1) * (original_max - original_min) + original_min
    
    return dequantized


def float_to_fixed_point(data: np.ndarray, integer_bits: int = 4, fractional_bits: int = 12) -> np.ndarray:
    """
    将浮点数据转换为定点表示
    
    参数:
    data: 输入浮点数据
    integer_bits: 整数部分位数
    fractional_bits: 小数部分位数
    
    返回:
    np.ndarray: 定点表示的数据
    """
    # 计算缩放因子
    scale = 2 ** fractional_bits
    
    # 转换为定点
    fixed_point = np.round(data * scale).astype(np.int32)
    
    # 检查是否超出范围
    total_bits = integer_bits + fractional_bits
    max_value = 2 ** (total_bits - 1) - 1
    min_value = -2 ** (total_bits - 1)
    
    # 裁剪超出范围的值
    fixed_point = np.clip(fixed_point, min_value, max_value)
    
    return fixed_point


def fixed_point_to_float(fixed_point: np.ndarray, fractional_bits: int = 12) -> np.ndarray:
    """
    将定点数据转换回浮点表示
    
    参数:
    fixed_point: 定点数据
    fractional_bits: 小数部分位数
    
    返回:
    np.ndarray: 浮点表示的数据
    """
    # 计算缩放因子
    scale = 2 ** fractional_bits
    
    # 转换为浮点
    float_data = fixed_point.astype(np.float32) / scale
    
    return float_data


def encode_matrix_for_hardware(matrix: np.ndarray, 
                              row_major: bool = True, 
                              conversion_scale: float = 1.0) -> np.ndarray:
    """
    将矩阵编码为硬件兼容的格式
    
    参数:
    matrix: 输入矩阵
    row_major: 是否以行优先顺序存储
    conversion_scale: 转换比例
    
    返回:
    np.ndarray: 编码后的一维数组
    """
    # 应用缩放
    scaled_matrix = matrix * conversion_scale
    
    # 按行或列展开
    if row_major:
        flattened = scaled_matrix.flatten()
    else:
        flattened = scaled_matrix.T.flatten()
    
    return flattened


def decode_matrix_from_hardware(data: np.ndarray, 
                               shape: Tuple[int, int], 
                               row_major: bool = True,
                               conversion_scale: float = 1.0) -> np.ndarray:
    """
    将硬件格式的数据解码为矩阵
    
    参数:
    data: 硬件数据
    shape: 矩阵形状 (rows, cols)
    row_major: 是否以行优先顺序存储
    conversion_scale: 转换比例
    
    返回:
    np.ndarray: 解码后的矩阵
    """
    rows, cols = shape
    
    # 重塑数据
    if row_major:
        matrix = data.reshape((rows, cols))
    else:
        matrix = data.reshape((cols, rows)).T
    
    # 应用反向缩放
    if conversion_scale != 0:
        matrix = matrix / conversion_scale
    
    return matrix


def encode_vector_for_hardware(vector: np.ndarray, conversion_scale: float = 1.0) -> np.ndarray:
    """
    将向量编码为硬件兼容的格式
    
    参数:
    vector: 输入向量
    conversion_scale: 转换比例
    
    返回:
    np.ndarray: 编码后的向量
    """
    # 确保是一维数组
    flattened = vector.flatten()
    
    # 应用缩放
    scaled_vector = flattened * conversion_scale
    
    return scaled_vector


def decode_vector_from_hardware(data: np.ndarray, conversion_scale: float = 1.0) -> np.ndarray:
    """
    将硬件格式的数据解码为向量
    
    参数:
    data: 硬件数据
    conversion_scale: 转换比例
    
    返回:
    np.ndarray: 解码后的向量
    """
    # 应用反向缩放
    if conversion_scale != 0:
        vector = data / conversion_scale
    else:
        vector = data.copy()
    
    return vector


def pack_data_for_transmission(data: Dict[str, Any]) -> bytes:
    """
    将数据打包为传输格式
    
    参数:
    data: 要打包的数据字典
    
    返回:
    bytes: 打包后的数据
    """
    import pickle
    import gzip
    
    # 使用pickle序列化
    serialized = pickle.dumps(data)
    
    # 使用gzip压缩
    compressed = gzip.compress(serialized)
    
    return compressed


def unpack_transmission_data(packed_data: bytes) -> Dict[str, Any]:
    """
    解包传输数据
    
    参数:
    packed_data: 打包的数据
    
    返回:
    Dict: 解包后的数据字典
    """
    import pickle
    import gzip
    
    # 解压缩
    decompressed = gzip.decompress(packed_data)
    
    # 反序列化
    data = pickle.loads(decompressed)
    
    return data


if __name__ == "__main__":
    # 简单测试
    print("数据转换测试")
    
    # 创建测试数据
    test_data = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
    
    # 测试归一化
    normalized = normalize_data(test_data, -1.0, 1.0)
    print(f"原始数据: {test_data}")
    print(f"归一化数据 [-1,1]: {normalized}")
    
    # 测试量化
    quantized = quantize_data(test_data, bits=4)
    print(f"量化数据 (4位): {quantized}")
    
    # 测试反量化
    dequantized = dequantize_data(quantized, np.min(test_data), np.max(test_data), bits=4)
    print(f"反量化数据: {dequantized}")
    
    # 测试定点转换
    fixed = float_to_fixed_point(test_data)
    print(f"定点数据: {fixed}")
    
    # 测试浮点转换
    restored = fixed_point_to_float(fixed)
    print(f"恢复的浮点数据: {restored}")
    
    # 测试矩阵编码/解码
    test_matrix = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    
    encoded = encode_matrix_for_hardware(test_matrix, conversion_scale=100)
    print(f"编码后的矩阵数据: {encoded}")
    
    decoded = decode_matrix_from_hardware(encoded, test_matrix.shape, conversion_scale=100)
    print(f"解码后的矩阵:\n{decoded}")
    
    # 测试数据打包/解包
    test_dict = {
        'vector': test_data,
        'matrix': test_matrix,
        'metadata': {'name': 'test', 'date': '2025-04-11'}
    }
    
    packed = pack_data_for_transmission(test_dict)
    print(f"打包数据大小: {len(packed)} 字节")
    
    unpacked = unpack_transmission_data(packed)
    print(f"解包数据键: {list(unpacked.keys())}")
    print(f"解包后的向量: {unpacked['vector']}") 