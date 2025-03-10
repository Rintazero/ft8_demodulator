import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

FT8_BAUD_RATE = 6.25 # symbols per second
FT8_SYMBOL_DURATION_S = 0.16  # 每个符号的持续时间（秒）
FT8_SYMBOL_FREQ_INTERVAL_HZ = 6.25  # 符号间的频率间隔（Hz）

SPECTROGRAM_BINS_PER_TONE = 10
SPECTROGRAM_STEPS_PER_SYMBOL = 10

FT8_NUM_SYNC_SEQUENCE = 3
FT8_NUM_SYNC_SYMBOLS_PER_SEQUENCE = 7
FT8_SYNC_PATTERN = [3, 1, 4, 0, 6, 5, 2]
FT8_SYNC_SEQUENCE_OFFSET = 36

def calculate_spectrogram(wave_data: np.ndarray, sample_rate: int, bins_per_tone: int = 2, steps_per_symbol: int = 2) -> tuple:
    """计算信号的频谱图
    
    Args:
        wave_data: 输入波形数据
        sample_rate: 采样率（Hz）
        bins_per_tone: 每个音调的频率bin数
        steps_per_symbol: 每个符号的时间步数
    
    Returns:
        tuple: (频谱图数据, 频率数组, 时间数组)
    """
    # 计算窗口参数
    samples_per_symbol = int(FT8_SYMBOL_DURATION_S * sample_rate)
    overlap_samples = samples_per_symbol - samples_per_symbol // steps_per_symbol
    dft_length = int(sample_rate / FT8_SYMBOL_FREQ_INTERVAL_HZ * bins_per_tone)
    
    # 检查信号长度是否足够
    if len(wave_data) < samples_per_symbol:
        # 如果信号太短，返回空的频谱图
        return np.array([[]]), np.array([]), np.array([])
    
    # 确保重叠长度小于窗口长度
    if overlap_samples >= samples_per_symbol:
        overlap_samples = samples_per_symbol - 1
    
    # 计算频谱图
    f,t,spectrogram = sci.signal.spectrogram(
        wave_data,
        fs=sample_rate,
        window='hann',
        nperseg=samples_per_symbol,
        noverlap=overlap_samples,
        nfft=dft_length,
        detrend=False,
        return_onesided=False,
        scaling='spectrum'
    )
    
    # 转换为dB单位
    with np.errstate(divide='ignore'):
        spectrogram = 10*np.log10(1E-12 + np.abs(spectrogram))

    # 将频率轴重新排序，使其从负频率到正频率
    spectrogram = np.fft.fftshift(spectrogram, axes=0)
    f = np.fft.fftshift(f)
    
    return spectrogram, f, t

def select_frequency_band(spectrogram: np.ndarray, f: np.ndarray, f_min: float, f_max: float) -> tuple:
    """选择频谱图中的特定频带
    
    Args:
        spectrogram: 频谱图数据
        f: 频率数组
        f_min: 最小频率（Hz）
        f_max: 最大频率（Hz）
    
    Returns:
        tuple: (过滤后的频谱图, 过滤后的频率数组)
    """
    # 找到频率范围内的索引
    mask = (f >= f_min) & (f <= f_max)
    return spectrogram[mask], f[mask]

