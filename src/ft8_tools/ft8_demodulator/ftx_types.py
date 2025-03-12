"""
FT8解调器的数据类型定义
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional

class FT8Protocol(Enum):
    """FT8 Protocol Type"""
    FT8 = 1

@dataclass
class FT8Waterfall:
    """Spectrogram waterfall data structure"""
    mag: np.ndarray  # Magnitude data, 2D array with shape (frequency, time)
    time_osr: int  # Time oversampling rate
    freq_osr: int  # Frequency oversampling rate
    
    def __post_init__(self):
        """Check if mag is a 2D array after initialization"""
        if len(self.mag.shape) != 2:
            raise ValueError("mag must be a 2D array with shape (frequency, time)")
    
    @property
    def num_bins(self) -> int:
        """Number of frequency bins (typically 8 for FT8)"""
        return self.mag.shape[0]  # FT8 uses 8-FSK modulation
    
    @property
    def num_blocks(self) -> int:
        """Number of time blocks"""
        return self.mag.shape[1] // self.time_osr
    

@dataclass
class FT8Candidate:
    """Candidate signal data structure"""
    waterfall: 'FT8Waterfall'  # Reference to the parent waterfall
    abs_time: int = 0  # Absolute time index
    abs_freq: int = 0  # Absolute frequency index
    score: float = 0.0

    def get_log_power(self, time_offset: int, freq_offset: int):
        """Get power at specified time and frequency offset"""
        return self.waterfall.mag[self.abs_freq + freq_offset * self.waterfall.freq_osr, self.abs_time + time_offset * self.waterfall.time_osr]

@dataclass
class FT8Message:
    """Decoded message data structure"""
    payload: bytearray = field(default_factory=lambda: bytearray(10))
    hash: int = 0

@dataclass
class FT8DecodeStatus:
    """Decode status data structure"""
    ldpc_errors: int = 0
    crc_extracted: int = 0
    crc_calculated: int = 0 
