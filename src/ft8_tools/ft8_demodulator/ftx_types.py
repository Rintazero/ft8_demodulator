"""
FT8解调器的数据类型定义
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

class FT8Protocol(Enum):
    """FT8协议类型"""
    FT8 = 1

@dataclass
class FT8Waterfall:
    """频谱瀑布数据结构"""
    mag: np.ndarray  # 幅度数据，形状为 (频率, 时间)的二维数组
    time_osr: int  # 时间过采样率
    freq_osr: int  # 频率过采样率
    
    def __post_init__(self):
        """初始化后检查mag是否为二维数组"""
        if len(self.mag.shape) != 2:
            raise ValueError("mag必须是二维数组，形状为(频率, 时间)")
    
    @property
    def num_bins(self) -> int:
        """频率bin数 (对于FT8通常是8)"""
        return self.mag.shape[0]  # FT8使用8-FSK调制
    
    @property
    def num_blocks(self) -> int:
        """时间块数"""
        return self.mag.shape[1] // self.time_osr
    

@dataclass
class FT8Candidate:
    """候选信号数据结构"""
    waterfall: 'FT8Waterfall'  # 对所属瀑布图的引用
    abs_time: int = 0  # 绝对时间索引
    abs_freq: int = 0  # 绝对频率索引
    score: float = 0.0

    def get_log_power(self, time_offset: int, freq_offset: int):
        """获取指定时间偏移和频率偏移的功率"""
        return self.waterfall.mag[self.abs_freq + freq_offset * self.waterfall.freq_osr, self.abs_time + time_offset * self.waterfall.time_osr]

@dataclass
class FT8Message:
    """解码后的消息数据结构"""
    payload: bytearray = bytearray(10)
    hash: int = 0

@dataclass
class FT8DecodeStatus:
    """解码状态数据结构"""
    ldpc_errors: int = 0
    crc_extracted: int = 0
    crc_calculated: int = 0 
