import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules
from ft8_tools.ft8_demodulator.spectrogram_analyse import (
    calculate_spectrogram,
    select_frequency_band,
)
from ft8_tools.ft8_demodulator.ft8_decode import (
    decode_ft8_message,
    FT8Message,
    FT8DecodeStatus,
)

from src.ft8_tools.ft8_demodulator.ftx_types import FT8Waterfall

# 导入本地模块
from src.ft8_tools.ft8_generator.modulator import ft8_generator, ft8_baseband_generator
from src.ft8_tools.ft8_beacon_receiver.frequency_correction import correct_frequency_drift
from src.ft8_tools.ft8_demodulator.spectrogram_analyse import (
    calculate_spectrogram,
    select_frequency_band,
)


workspace_path = "./src/tests/channel/doppler_shift_test"



import scipy
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
def gfsk_pulse(bt, t):
    """
    生成GFSK脉冲
    
    参数:
    bt: 带宽时间乘积
    t: 时间序列
    
    返回:
    output: GFSK脉冲
    """
    k = np.pi * np.sqrt(2.0/np.log(2.0))
    output = 0.5 * (scipy.special.erf(k*bt*(t+0.5)) - scipy.special.erf(k*bt*(t-0.5)))
    return output


# 加载信号
down_sampled_signal_path = os.path.join(workspace_path, 'down_sampled_signal.npy')
if not os.path.exists(down_sampled_signal_path):
    raise ValueError(f"down_sampled_signal.npy does not exist in {workspace_path}")
else:
    down_sampled_signal = np.load(down_sampled_signal_path)

# 加载 Info
info_path = os.path.join(workspace_path, 'signal_processing_info.txt')
if not os.path.exists(info_path):
    raise ValueError(f"signal_processing_info.txt does not exist in {workspace_path}")
else:
    with open(info_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'down_sample_factor:' in line:
                down_sample_factor = int(line.split(':')[1].strip())
            elif 'fs_Hz:' in line:
                fs_Hz = float(line.split(':')[1].strip())

print(f"down_sample_factor: {down_sample_factor}")
print(f"fs_Hz: {fs_Hz}")

# 尝试解调

## Basic Parameters
bins_per_tone = 2
steps_per_symbol = 2
mask_f_Max = 300
mask_f_Min = 0

## 计算频谱图
spectrogram, f, t = calculate_spectrogram(
    down_sampled_signal, fs_Hz, bins_per_tone, steps_per_symbol
)

### 根据 mask_f_Max 和 mask_f_Min 过滤频谱图
freq_mask = (f >= mask_f_Min) & (f <= mask_f_Max)
spectrogram = spectrogram[freq_mask]
f = f[freq_mask]

### 绘制频谱图
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, t[-1], f[0], f[-1]])
plt.colorbar(label='Intensity (dB)')
plt.show()
plt.savefig('spectrogram.png')


# 创建FT8Waterfall对象
waterfall = FT8Waterfall(
    mag=spectrogram,
    time_osr=2,
    freq_osr=2
)

# 应用频率校正 - 直接使用复信号进行校正
print("Applying frequency correction...")
wave_corrected, estimated_drift_rate = correct_frequency_drift(
    down_sampled_signal, 
    fs_Hz, 
    bins_per_tone, 
    steps_per_symbol,
    waterfall,
    params={
        'nsync_sym': 7,
        'ndata_sym': 58,
        'zscore_threshold': 5,
        'max_iteration_num': 400000,
        'debug_plots': False
    }
)

print(f"Estimated frequency drift rate: {estimated_drift_rate} Hz/sample")


## signal detection
__detection_method__ = 'time_domain_correlation_with_gfsk'

if __detection_method__ == 'time_domain_correlation':
    ### Calculate maximum amplitude frequency-time sequence
    windowSum = np.zeros(spectrogram.shape)
    for i in range(spectrogram.shape[0]):
        for j in range(spectrogram.shape[1]):
            if i < spectrogram.shape[0] - bins_per_tone:
                windowSum[i][j] = np.sum(spectrogram[i:i+bins_per_tone,j])
            else:
                windowSum[i][j] = np.sum(spectrogram[i:,j])
    windowIndices = np.argmax(windowSum, axis=0)
    max_freq_indices = np.zeros(spectrogram.shape[1])
    for i in range(spectrogram.shape[1]):
        max_freq_indices[i] = windowIndices[i] + np.argmax(spectrogram[windowIndices[i]:windowIndices[i]+bins_per_tone,i])
    max_freq_indices = max_freq_indices.astype(int)
    max_freqs = f[max_freq_indices]

    plt.figure(num=0, figsize=(10, 6))
    plt.plot(np.linspace(0, t[-1], len(max_freqs)), max_freqs, marker='o', linestyle='-', color='blue', label='Original Data')
    plt.title('Maximum Frequency vs Time (Before Processing)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, t[-1])
    plt.ylim(f[0], f[-1])
    plt.savefig('max_frequencies_before_processing.png')
    plt.show()

    ### 构造同步序列
    SyncSymSeq = (np.array([3,1,4,0,6,5,2]))
    SyncSymSeq = SyncSymSeq - np.mean(SyncSymSeq)
    SyncSeq = np.zeros(len(SyncSymSeq) * steps_per_symbol)
    for i in range(len(SyncSymSeq)):
        SyncSeq[i*steps_per_symbol] = SyncSymSeq[i]
        SyncSeq[i*steps_per_symbol+1] = SyncSymSeq[i]

    ### 计算同步序列相关
    syncCorrelation = np.correlate(max_freqs, SyncSeq, mode='full')

    ### 绘制同步序列相关
    plt.figure(num=1, figsize=(10, 6))
    plt.plot(np.linspace(0, t[-1], len(syncCorrelation)), syncCorrelation, marker='o', linestyle='-', color='blue', label='Sync Correlation')
    plt.title('Sync Correlation')
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.legend()
    plt.savefig('sync_correlation.png')
    plt.show()

    ### 计算同步序列相关峰值
    syncCorrelationPeak = np.max(syncCorrelation)

if __detection_method__ == 'time_domain_correlation_with_gfsk':
    ### Calculate maximum amplitude frequency-time sequence
    windowSum = np.zeros(spectrogram.shape)
    for i in range(spectrogram.shape[0]):
        for j in range(spectrogram.shape[1]):
            if i < spectrogram.shape[0] - bins_per_tone:
                windowSum[i][j] = np.sum(spectrogram[i:i+bins_per_tone,j])
            else:
                windowSum[i][j] = np.sum(spectrogram[i:,j])
    windowIndices = np.argmax(windowSum, axis=0)
    max_freq_indices = np.zeros(spectrogram.shape[1])
    for i in range(spectrogram.shape[1]):
        max_freq_indices[i] = windowIndices[i] + np.argmax(spectrogram[windowIndices[i]:windowIndices[i]+bins_per_tone,i])
    max_freq_indices = max_freq_indices.astype(int)
    max_freqs = f[max_freq_indices]

       # 构造同步序列
    sync_seq = (np.array([3, 1, 4, 0, 6, 5, 2]) + 1)
    sync_seq = sync_seq - np.mean(sync_seq)

    # 每个符号的GFSK脉冲整形
    samples_per_sym = steps_per_symbol * 2
    t_pulse = np.linspace(-1, 1, samples_per_sym+1)
    gfsk_shape = gfsk_pulse(bt=2.0, t=t_pulse)

    # 扩展同步序列长度以适应整形脉冲
    sync_correlation_seq = np.zeros((7-1) * steps_per_symbol + samples_per_sym + 1)

    # 对每个同步符号进行脉冲整形
    for sym_idx in range(7):
        sync_correlation_seq[sym_idx * steps_per_symbol:(sym_idx * steps_per_symbol) + samples_per_sym + 1] += gfsk_shape * sync_seq[sym_idx]

    # 创建三个同步序列
    three_sync_correlation_seq = np.zeros((3*7 + 58 - 1) * steps_per_symbol + 1 + samples_per_sym)

    for i in range(3):
        start_idx = i*(7+58//2)*steps_per_symbol
        end_idx = start_idx + len(sync_correlation_seq)
        three_sync_correlation_seq[start_idx:end_idx] = sync_correlation_seq
    
    ### 计算同步块相关
    # syncCorrelation = np.correlate(max_freqs, three_sync_correlation_seq, mode='full')
    ### 计算同步块相关
    syncCorrelation = np.correlate(max_freqs, three_sync_correlation_seq, mode='full')

    ### 绘制同步块相关
    plt.figure(num=1, figsize=(10, 6))
    plt.plot(np.linspace(0, t[-1], len(syncCorrelation)), syncCorrelation, marker='o', linestyle='-', color='blue', label='Sync Correlation')
    plt.title('Sync Correlation')
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.legend()
    # plt.savefig('sync_correlation.png')
    plt.show()

if __detection_method__ == 'time_frequency_block_correlation':

    ### 构造同步序列
    SyncSymSeq = (np.array([3,1,4,0,6,5,2]))
    SyncSymSeq = SyncSymSeq - np.mean(SyncSymSeq)
    SyncSeq = np.zeros(len(SyncSymSeq) * steps_per_symbol)
    for i in range(len(SyncSymSeq)):
        SyncSeq[i*steps_per_symbol] = SyncSymSeq[i]
        SyncSeq[i*steps_per_symbol+1] = SyncSymSeq[i]

    

    ### 绘制同步块相关
    plt.figure(num=1, figsize=(10, 6))