"""
内存管理模块

该模块负责管理阻变存储器的内存分配和释放，跟踪数据位置，并提供内存映射功能。
"""

import os
import time
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Any
import logging


class MemoryManager:
    """管理阻变存储器的内存资源"""
    
    def __init__(self, max_memory_size: int = 4096, block_size: int = 16):
        """
        初始化内存管理器
        
        参数:
        max_memory_size: 最大内存大小
        block_size: 内存块大小
        """
        self.max_memory_size = max_memory_size
        self.block_size = block_size
        
        # 计算块数量
        self.num_blocks = max_memory_size // block_size
        
        # 初始化内存分配表
        # 0: 未分配，1: 已分配
        self.allocation_table = np.zeros(self.num_blocks, dtype=np.int8)
        
        # 跟踪已分配的内存
        # 键: 分配ID, 值: (起始地址, 大小)
        self.allocations = {}
        
        # 内存映射 - 跟踪数据存储位置
        # 键: 数据ID, 值: 分配ID
        self.memory_map = {}
        
        # 设置日志记录器
        self.logger = logging.getLogger('MemoryManager')
        self.setup_logger()
    
    def setup_logger(self) -> None:
        """设置日志记录器"""
        self.logger.setLevel(logging.INFO)
        
        # 检查是否已有处理器，如果没有则添加
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def allocate_memory(self, size: int) -> Optional[int]:
        """
        分配内存
        
        参数:
        size: 请求的内存大小
        
        返回:
        Optional[int]: 分配的起始地址，失败则返回None
        """
        # 计算需要的块数量
        num_blocks_needed = (size + self.block_size - 1) // self.block_size
        
        # 检查是否有足够的空间
        if num_blocks_needed > self.num_blocks:
            self.logger.error(f"请求的内存 ({size} 字节) 超过最大内存大小 ({self.max_memory_size} 字节)")
            return None
        
        # 查找连续的空闲块
        start_block = self._find_free_blocks(num_blocks_needed)
        
        if start_block is None:
            self.logger.error(f"无法找到 {num_blocks_needed} 个连续的空闲块")
            return None
        
        # 计算起始地址
        start_address = start_block * self.block_size
        
        # 标记块为已分配
        for i in range(num_blocks_needed):
            self.allocation_table[start_block + i] = 1
        
        # 生成分配ID
        allocation_id = self._generate_allocation_id()
        
        # 记录分配
        self.allocations[allocation_id] = (start_address, size)
        
        self.logger.info(f"已分配 {size} 字节内存，起始地址: {start_address}, 分配ID: {allocation_id}")
        
        return start_address
    
    def free_memory(self, address: int) -> bool:
        """
        释放内存
        
        参数:
        address: 内存起始地址
        
        返回:
        bool: 是否成功释放
        """
        # 查找对应的分配ID
        allocation_id = None
        for aid, (addr, _) in self.allocations.items():
            if addr == address:
                allocation_id = aid
                break
        
        if allocation_id is None:
            self.logger.error(f"找不到地址 {address} 的内存分配")
            return False
        
        return self.free_memory_by_id(allocation_id)
    
    def free_memory_by_id(self, allocation_id: str) -> bool:
        """
        通过分配ID释放内存
        
        参数:
        allocation_id: 分配ID
        
        返回:
        bool: 是否成功释放
        """
        if allocation_id not in self.allocations:
            self.logger.error(f"找不到分配ID {allocation_id}")
            return False
        
        # 获取分配信息
        start_address, size = self.allocations[allocation_id]
        
        # 计算块信息
        start_block = start_address // self.block_size
        num_blocks = (size + self.block_size - 1) // self.block_size
        
        # 标记块为未分配
        for i in range(num_blocks):
            self.allocation_table[start_block + i] = 0
        
        # 移除分配记录
        del self.allocations[allocation_id]
        
        # 更新内存映射
        data_ids_to_remove = []
        for data_id, aid in self.memory_map.items():
            if aid == allocation_id:
                data_ids_to_remove.append(data_id)
        
        for data_id in data_ids_to_remove:
            del self.memory_map[data_id]
        
        self.logger.info(f"已释放 {size} 字节内存，起始地址: {start_address}, 分配ID: {allocation_id}")
        
        return True
    
    def register_data(self, data_id: str, size: int) -> Optional[int]:
        """
        注册数据并分配内存
        
        参数:
        data_id: 数据ID
        size: 数据大小
        
        返回:
        Optional[int]: 分配的起始地址，失败则返回None
        """
        # 如果数据ID已存在，先释放旧的内存
        if data_id in self.memory_map:
            old_allocation_id = self.memory_map[data_id]
            self.free_memory_by_id(old_allocation_id)
        
        # 分配新内存
        address = self.allocate_memory(size)
        
        if address is None:
            return None
        
        # 查找对应的分配ID
        allocation_id = None
        for aid, (addr, _) in self.allocations.items():
            if addr == address:
                allocation_id = aid
                break
        
        if allocation_id is None:
            self.logger.error("内部错误: 无法找到刚刚分配的内存")
            return None
        
        # 更新内存映射
        self.memory_map[data_id] = allocation_id
        
        return address
    
    def get_data_address(self, data_id: str) -> Optional[int]:
        """
        获取数据的内存地址
        
        参数:
        data_id: 数据ID
        
        返回:
        Optional[int]: 数据的内存地址，如果未找到则返回None
        """
        if data_id not in self.memory_map:
            self.logger.error(f"找不到数据ID {data_id}")
            return None
        
        allocation_id = self.memory_map[data_id]
        
        if allocation_id not in self.allocations:
            self.logger.error(f"内部错误: 找不到分配ID {allocation_id}")
            return None
        
        address, _ = self.allocations[allocation_id]
        return address
    
    def get_memory_utilization(self) -> Dict[str, Any]:
        """
        获取内存利用率信息
        
        返回:
        Dict: 内存利用率信息
        """
        used_blocks = np.sum(self.allocation_table)
        total_blocks = len(self.allocation_table)
        
        # 计算已分配和空闲内存
        allocated_memory = used_blocks * self.block_size
        free_memory = (total_blocks - used_blocks) * self.block_size
        
        # 计算利用率
        utilization = (used_blocks / total_blocks) * 100 if total_blocks > 0 else 0
        
        # 收集分配信息
        allocations_info = []
        for allocation_id, (address, size) in self.allocations.items():
            # 查找使用此分配的数据
            data_ids = []
            for data_id, aid in self.memory_map.items():
                if aid == allocation_id:
                    data_ids.append(data_id)
            
            allocations_info.append({
                'id': allocation_id,
                'address': address,
                'size': size,
                'data_ids': data_ids
            })
        
        return {
            'total_memory': self.max_memory_size,
            'allocated_memory': allocated_memory,
            'free_memory': free_memory,
            'utilization': utilization,
            'total_blocks': total_blocks,
            'used_blocks': used_blocks,
            'free_blocks': total_blocks - used_blocks,
            'allocations': allocations_info
        }
    
    def defragment(self) -> None:
        """
        对内存进行碎片整理
        
        注意: 这仅仅是内存管理的碎片整理，实际数据移动需要由外部处理
        """
        # 创建一个新的管理器
        new_manager = MemoryManager(self.max_memory_size, self.block_size)
        
        # 按大小排序分配，以减少碎片
        sorted_allocations = sorted(
            self.allocations.items(), 
            key=lambda x: x[1][1],  # 按大小排序
            reverse=True
        )
        
        # 数据ID到新地址的映射
        data_address_map = {}
        
        # 重新分配内存
        for allocation_id, (old_address, size) in sorted_allocations:
            # 分配新内存
            new_address = new_manager.allocate_memory(size)
            
            if new_address is None:
                self.logger.error(f"碎片整理时分配内存失败: {allocation_id}, 大小: {size}")
                continue
            
            # 找到使用此分配的数据
            for data_id, aid in self.memory_map.items():
                if aid == allocation_id:
                    # 更新数据地址映射
                    data_address_map[data_id] = {
                        'old_address': old_address,
                        'new_address': new_address,
                        'size': size
                    }
        
        # 替换当前的内存分配
        self.allocation_table = new_manager.allocation_table.copy()
        self.allocations = new_manager.allocations.copy()
        
        # 更新内存映射
        new_memory_map = {}
        for data_id, allocation_id in self.memory_map.items():
            # 查找新的分配ID
            new_allocation_id = None
            for new_aid, (new_addr, _) in new_manager.allocations.items():
                if new_addr == data_address_map[data_id]['new_address']:
                    new_allocation_id = new_aid
                    break
            
            if new_allocation_id is not None:
                new_memory_map[data_id] = new_allocation_id
        
        self.memory_map = new_memory_map
        
        self.logger.info(f"内存碎片整理完成, 需要移动 {len(data_address_map)} 个数据块")
        
        # 返回数据移动映射，以便外部处理实际数据移动
        return data_address_map
    
    def _find_free_blocks(self, num_blocks: int) -> Optional[int]:
        """
        查找连续的空闲块
        
        参数:
        num_blocks: 需要的块数量
        
        返回:
        Optional[int]: 找到的第一个空闲块的索引，如果未找到则返回None
        """
        current_free = 0
        start_block = None
        
        for i in range(self.num_blocks):
            if self.allocation_table[i] == 0:  # 空闲块
                if start_block is None:
                    start_block = i
                current_free += 1
                
                if current_free >= num_blocks:
                    return start_block
            else:  # 已分配块
                start_block = None
                current_free = 0
        
        return None
    
    def _generate_allocation_id(self) -> str:
        """
        生成唯一的分配ID
        
        返回:
        str: 分配ID
        """
        timestamp = int(time.time() * 1000)
        random_part = np.random.randint(0, 1000000)
        return f"alloc_{timestamp}_{random_part}"


if __name__ == "__main__":
    # 简单测试
    print("内存管理器测试")
    
    # 创建一个256字节内存的管理器，块大小为16字节
    manager = MemoryManager(256, 16)
    
    # 分配一些内存
    addr1 = manager.register_data("matrix_A", 32)
    addr2 = manager.register_data("vector_B", 16)
    addr3 = manager.register_data("result_C", 48)
    
    print(f"matrix_A 地址: {addr1}")
    print(f"vector_B 地址: {addr2}")
    print(f"result_C 地址: {addr3}")
    
    # 获取内存利用率
    utilization = manager.get_memory_utilization()
    
    print(f"\n内存利用率: {utilization['utilization']:.2f}%")
    print(f"已分配: {utilization['allocated_memory']} 字节")
    print(f"空闲: {utilization['free_memory']} 字节")
    
    # 获取数据地址
    matrix_addr = manager.get_data_address("matrix_A")
    print(f"\nmatrix_A 的内存地址: {matrix_addr}")
    
    # 释放一些内存
    print("\n释放 vector_B")
    manager.free_memory(addr2)
    
    # 重新分配
    addr4 = manager.register_data("vector_D", 64)
    print(f"vector_D 地址: {addr4}")
    
    # 进行碎片整理
    print("\n进行碎片整理")
    data_moves = manager.defragment()
    
    print("需要移动的数据:")
    for data_id, move_info in data_moves.items():
        print(f"  {data_id}: {move_info['old_address']} -> {move_info['new_address']}")
    
    # 最终内存利用率
    utilization = manager.get_memory_utilization()
    print(f"\n碎片整理后内存利用率: {utilization['utilization']:.2f}%")
    
    # 打印所有分配
    print("\n当前分配:")
    for alloc in utilization['allocations']:
        print(f"  {alloc['id']}: 地址={alloc['address']}, 大小={alloc['size']}, 数据={alloc['data_ids']}") 