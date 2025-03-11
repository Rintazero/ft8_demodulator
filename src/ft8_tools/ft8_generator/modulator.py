import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import math
from . import encoder
from numpy.typing import NDArray
from .encoder import FT8_SYMBOL_NUM

FT8_SYMBOL_TIME_S = 0.16 # 符号时间
FT8_SYMBOL_FREQ_INTERVAL_HZ = 6.25 # 符号频率间隔

# SAMPLE_RATE_HZ = 10e3 # 采样率
# CENTER_FREQ_HZ = 550 # 中心频率

# sample_per_symbol = int(FT8_SYMBOL_TIME_S * SAMPLE_RATE_HZ) # 每个符号的采样点数
# dt = 1 / SAMPLE_RATE_HZ # 采样间隔
# dphi_peak = 2 * np.pi * FT8_SYMBOL_FREQ_INTERVAL_HZ / SAMPLE_RATE_HZ # 峰值相位偏移
# df_peak = FT8_SYMBOL_FREQ_INTERVAL_HZ # 峰值频率偏移

def gauss_window_generator(bt: float, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Generate the Gaussian window.
    """
    k = np.pi * np.sqrt(2 / np.log(2))
    return 0.5 * (sci.special.erf(k*bt*(t+0.5)) - sci.special.erf(k*bt*(t-0.5)))

def gfsk_modulation_waveform_generator(itones: NDArray[np.uint8],fs: float) -> NDArray[np.float64]:
    """
    Generate the waveform for FT8.  
    """
    sample_per_symbol = int(FT8_SYMBOL_TIME_S * fs) # 每个符号的采样点数
    df_peak = FT8_SYMBOL_FREQ_INTERVAL_HZ # 峰值频率偏移

    t = (np.arange(3*sample_per_symbol)-1.5*sample_per_symbol)/sample_per_symbol
    window = gauss_window_generator(2.0, t)

    # 计算频率序列
    freq_seq = np.zeros((FT8_SYMBOL_NUM+2)*sample_per_symbol, dtype=np.float64)
    
    for i in range(FT8_SYMBOL_NUM):
        ib = i*sample_per_symbol
        for j in range(3*sample_per_symbol):
            freq_seq[ib+j] += df_peak * itones[i] * window[j]
    for i in range(2*sample_per_symbol):
        freq_seq[i] += df_peak * itones[0] * window[i + sample_per_symbol]
        freq_seq[i+FT8_SYMBOL_NUM*sample_per_symbol] += df_peak * itones[FT8_SYMBOL_NUM-1] * window[i]

    return freq_seq

def freq_to_dphi(freq: NDArray[np.float64],fs: float) -> NDArray[np.float64]:
    """
    Convert frequency to phase.
    """
    return 2 * np.pi * freq / fs

def ft8_modulation_waveform_generator(gfsk_waveform: NDArray[np.float64],fs: float,f0: float) -> NDArray[np.complex128]:
    """
    Generate the waveform for FT8.  
    """
    sample_per_symbol = int(FT8_SYMBOL_TIME_S * fs) # 每个符号的采样点数
    dphi = freq_to_dphi(gfsk_waveform,fs) + 2 * np.pi * f0 / fs
    phi = 0

    ft8_waveform = np.zeros(FT8_SYMBOL_NUM * sample_per_symbol, dtype=np.complex128)

    for i in range(FT8_SYMBOL_NUM * sample_per_symbol):
        ft8_waveform[i] = np.sin(phi) - 1j*np.cos(phi)
        phi = math.fmod(phi + dphi[i], 2 * np.pi)

    nramp = sample_per_symbol // 8
    for i in range(nramp):
        ft8_waveform[i] *= 0.5 * (1 - np.cos(8 * np.pi * i / sample_per_symbol))
        ft8_waveform[FT8_SYMBOL_NUM * sample_per_symbol - i - 1] *= 0.5 * (1 + np.cos(8 * np.pi * i / sample_per_symbol))

    return ft8_waveform

def ft8_baseband_generator(payload: NDArray[np.uint8],fs: float,f0: float) -> NDArray[np.complex128]:
    """
    Generate the baseband waveform for FT8.  
    """
    itones = encoder.ft8_encode(payload)
    gfsk_waveform = gfsk_modulation_waveform_generator(itones,fs)
    return ft8_modulation_waveform_generator(gfsk_waveform,fs,f0)

def ft8_generator(payload: NDArray[np.uint8],fs: float,f0: float,fc: float) -> NDArray[np.float64]:
    """
    Generate the waveform for FT8.  
    """
    ft8_baseband = ft8_baseband_generator(payload,fs,f0)
    return np.real(ft8_baseband*np.exp(1j*2*np.pi*fc*np.arange(len(ft8_baseband))/fs))
