import sys
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ft8_tools.channel.channel import Channel
from ft8_tools.ft8_generator import ft8_baseband_generator

fs_Hz = 50e3

workspace_path = "./src/tests/channel/doppler_shift_test"
os.makedirs(workspace_path, exist_ok=True)

# 加载信号
signal_path = os.path.join(workspace_path, 'signal_with_doppler_shift.npy')
if not os.path.exists(signal_path):
    raise ValueError(f"signal_with_doppler_shift.npy does not exist in {workspace_path}")
else:
    signal = np.load(signal_path)

# 加载线性回归结果
linear_regression_path = os.path.join(workspace_path, 'doppler_frequency_shift_info.txt')
if not os.path.exists(linear_regression_path):
    raise ValueError(f"doppler_frequency_shift_info.txt does not exist in {workspace_path}")
else:
    with open(linear_regression_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'Slope:' in line:
                slope = float(line.split(':')[1].strip())
            elif 'Intercept:' in line:
                intercept = float(line.split(':')[1].strip())

print(f"Slope(Hz/sample): {slope}")
print(f"Intercept(Hz): {intercept}")

# 计算补偿后的信号
num_samples = len(signal)
time_vector = np.arange(num_samples) / fs_Hz # 生成时间向量

# 计算多普勒频偏补偿
doppler_shift_correction = np.exp(1j * 2 * np.pi * (slope * time_vector * fs_Hz + intercept) * time_vector)
compensated_signal = signal * doppler_shift_correction

down_sample_factor = 25
down_sampled_signal = compensated_signal[::down_sample_factor]

# 保存补偿后的信号
np.save(os.path.join(workspace_path, 'compensated_signal.npy'), compensated_signal)

# 保存降采样之后的信号
np.save(os.path.join(workspace_path, 'down_sampled_signal.npy'), down_sampled_signal)

os.makedirs(workspace_path, exist_ok=True)
info_path = os.path.join(workspace_path, 'signal_processing_info.txt')
with open(info_path, 'w') as f:
    f.write("Signal Processing Info\n")
    f.write("----------------------------------\n")
    f.write("compensated_signal Info\n")
    f.write(f"fs_Hz: {fs_Hz}\n")
    f.write("----------------------------------\n")
    f.write("down_sampled_signal Info\n")
    f.write(f"down_sample_factor: {down_sample_factor}\n")
    f.write(f"fs_Hz: {fs_Hz / down_sample_factor}\n")

# 绘制频谱图
plt.figure(figsize=(12, 8))

# 绘制补偿后信号的频谱图
plt.subplot(2, 1, 1)
plt.specgram(compensated_signal, NFFT=256, Fs=fs_Hz, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.title('Spectrogram of Compensated Signal')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude (dB)')
plt.grid()

# 绘制降采样信号的频谱图
plt.subplot(2, 1, 2)
plt.specgram(down_sampled_signal, NFFT=256, Fs=fs_Hz / down_sample_factor, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.title('Spectrogram of Downsampled Signal')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude (dB)')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(workspace_path, 'signal(downsampled)_spectrograms.png'))








