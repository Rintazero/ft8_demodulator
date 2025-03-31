import datetime
import numpy
import sys
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import scipy.io.wavfile as wavfile

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ft8_tools.channel.channel import Channel
from ft8_tools.ft8_generator import ft8_baseband_generator

fc_Hz = 2.45e9
fs_Hz = 50e3
f0_Hz = 100
SignalTime_s = 20
SignalTimeShift_s = 3

# 创建保存目录
save_path = "./src/tests/channel/doppler_shift_test"
os.makedirs(save_path, exist_ok=True)

# 检查多普勒频移数据文件是否存在
load_doppler_shift_path = os.path.join(save_path, 'doppler_frequency_shift.npy')
if not os.path.exists(load_doppler_shift_path):
    raise ValueError(f"doppler_frequency_shift.npy does not exist in {save_path}")
else:
    doppler_shift_Hz_seq = np.load(load_doppler_shift_path)

# generate baseband signal
payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
baseband_signal = ft8_baseband_generator(payload, fs_Hz, f0_Hz)

baseband_power = np.mean(np.abs(baseband_signal)**2)
SNR_dB = 10  # Signal-to-Noise Ratio in dB
SNR_linear = 10 ** (SNR_dB / 10)  # Convert SNR from dB to linear scale
noise_power = baseband_power / SNR_linear  # Calculate noise power

num_samples = int(SignalTime_s * fs_Hz)
gaussian_noise = np.random.normal(0, np.sqrt(noise_power), num_samples) + 1j * np.random.normal(0, np.sqrt(noise_power), num_samples)

SignalTimeShift_samples = int(SignalTimeShift_s * fs_Hz)  # 转换为采样点数
extended_baseband_signal = np.zeros(num_samples, dtype=np.complex128)
extended_baseband_signal[SignalTimeShift_samples:SignalTimeShift_samples + len(baseband_signal)] = baseband_signal

# Apply Doppler shift to the baseband signal
num_samples = len(extended_baseband_signal)
doppler_shifted_signal = np.zeros(num_samples, dtype=np.complex128)

print(f"len(extended_baseband_signal): {len(extended_baseband_signal)}; len(doppler_shift_Hz_seq): {len(doppler_shift_Hz_seq)}\n")

for i in range(num_samples):
    current_time = i / fs_Hz
    doppler_shift = doppler_shift_Hz_seq[i] if i < len(doppler_shift_Hz_seq) else 0
    doppler_phase_shift = np.exp(-1j * 2 * np.pi * doppler_shift * current_time)
    doppler_shifted_signal[i] = extended_baseband_signal[i] * doppler_phase_shift + gaussian_noise[i]

signal = doppler_shifted_signal

# 保存复数信号
np.save(os.path.join(save_path, 'signal_with_doppler_shift.npy'), signal)

# 绘制频谱图
plt.figure(figsize=(12, 8))

# 添加一个小的常数来避免log10(0)
eps = 1e-10

# 绘制原始基带信号的频谱图
plt.subplot(2, 1, 1)
plt.specgram(extended_baseband_signal + eps, NFFT=256, Fs=fs_Hz, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.title('Spectrogram of Original Baseband Signal')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude (dB)')
plt.grid()

# 绘制带有多普勒频移和噪声的信号的频谱图
plt.subplot(2, 1, 2)
plt.specgram(signal + eps, NFFT=256, Fs=fs_Hz, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.title('Spectrogram of Signal with Doppler Shift and Noise')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude (dB)')
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'signal(baseband and with doppler shift)_spectrograms.png'))










