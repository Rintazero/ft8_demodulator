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

workspace_path = "./src/tests/channel/doppler_shift_test"

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

## signal detection
__detection_method__ = 'time_domain_correlation'

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
    # plt.savefig('sync_correlation.png')
    plt.show()

    ### 计算同步序列相关峰值
    syncCorrelationPeak = np.max(syncCorrelation)

if __detection_method__ == 'time_frequency_block_correlation':
    ### 构造同步块
    SyncSymSeq = (np.array([3,1,4,0,6,5,2]))
    SyncSymSeq = SyncSymSeq - np.mean(SyncSymSeq)
    tSyncSeq = np.zeros(len(SyncSymSeq) * steps_per_symbol)
    for i in range(len(SyncSymSeq)):
        tSyncSeq[i*steps_per_symbol] = SyncSymSeq[i]
        tSyncSeq[i*steps_per_symbol+1] = SyncSymSeq[i]

    ### 计算同步块相关
    syncCorrelation = np.correlate(max_freqs, tSyncSeq, mode='full')

    ### 绘制同步块相关
    plt.figure(num=1, figsize=(10, 6))