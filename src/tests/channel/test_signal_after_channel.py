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
from ft8_tools.ft8_beacon_receiver.frequency_correction import correct_frequency_drift
from ft8_tools.ft8_demodulator.ft8_decode import decode_ft8_message, FT8Message, FT8DecodeStatus

fc_Hz = 2.405e9
fs_Hz = 50e3
f0_Hz = 100 + 3.125
SignalTime_s = 20
SignalTimeShift_s = 1

# 创建保存目录
save_path = "./src/tests/channel/doppler_shift_test"
os.makedirs(save_path, exist_ok=True)

# 检查多普勒频移数据文件是否存在
load_doppler_shift_path = os.path.join(save_path, 'doppler_frequency_shift.npy')
if not os.path.exists(load_doppler_shift_path):
    raise ValueError(f"doppler_frequency_shift.npy does not exist in {save_path}")
else:
    doppler_shift_Hz_seq = np.load(load_doppler_shift_path)

# 绘制多普勒频移序列
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(doppler_shift_Hz_seq))/fs_Hz, doppler_shift_Hz_seq)
plt.xlabel('时间 (s)')
plt.ylabel('多普勒频移 (Hz)')
plt.title('多普勒频移随时间的变化')
plt.grid(True)
plt.savefig(os.path.join(save_path, 'doppler_shift.png'))
# plt.show()
plt.close()

# generate baseband signal
payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x50], dtype=np.uint8)
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

# 初始化相位
theta = 0

for i in range(num_samples):
    # 累加相位
    if i < len(doppler_shift_Hz_seq):
        theta += 2 * np.pi * doppler_shift_Hz_seq[i] / fs_Hz
    
    # 应用相位偏移
    doppler_phase_shift = np.exp(-1j * theta)
    doppler_shifted_signal[i] = extended_baseband_signal[i] * doppler_phase_shift + gaussian_noise[i]

signal = doppler_shifted_signal

# 保存复数信号
np.save(os.path.join(save_path, 'signal_with_doppler_shift.npy'), signal)
# np.load(os.path.join(save_path, 'signal_with_doppler_shift.npy'))

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
# plt.show()
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'signal(baseband and with doppler shift)_spectrograms.png'))
plt.close()

# 进行频率校正
print("开始进行频率校正...")
correction_params = {
    'nsync_sym': 7,
    'ndata_sym': 58,
    'zscore_threshold': 5,
    'max_iteration_num': 400000,
    'bins_per_tone': 2,
    'steps_per_symbol': 8,
    'precise_sync': True,  # 是否进行精确时间同步
    'poly_degree': 2,
    
}

# 应用频率校正
wave_corrected, estimated_drift_rate = correct_frequency_drift(
    signal,
    fs_Hz,
    6.25,  # sym_bin - FT8符号频率间隔
    0.16,  # sym_t - FT8符号时间长度
    params=correction_params
)

# wave_corrected2, estimated_drift_rate2 = correct_frequency_drift(
#     wave_corrected,
#     fs_Hz,
#     6.25,  # sym_bin - FT8符号频率间隔
#     0.16,  # sym_t - FT8符号时间长度
#     params=correction_params
# )

print(f"估计的频率漂移率: {estimated_drift_rate} Hz/sample")

# 计算校正后信号的频谱图
plt.figure(figsize=(10, 6))
plt.specgram(wave_corrected + eps, NFFT=256, Fs=fs_Hz, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.title('Spectrogram of Frequency Corrected Signal')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude (dB)')
plt.grid()
plt.savefig(os.path.join(save_path, 'corrected_signal_spectrogram.png'))
plt.close()

# 解码校正后的信号
print("开始解码校正后的信号...")
decode_params = {
    'bins_per_tone': 4,
    'steps_per_symbol': 4,
    'max_candidates': 100,
    'min_score': 6,
    'max_iterations': 40,
    'time_min': SignalTimeShift_s,
    'time_max': None,

}

results_after_correction = decode_ft8_message(
    wave_corrected,
    sample_rate=fs_Hz,
    bins_per_tone=decode_params['bins_per_tone'],
    steps_per_symbol=decode_params['steps_per_symbol'],
    max_candidates=decode_params['max_candidates'],
    min_score=decode_params['min_score'],
    max_iterations=decode_params['max_iterations'],
    freq_max=-7000,
    freq_min=-10000

)

# 输出解码结果
if results_after_correction:
    print("频率校正后成功解码!")
    decoded_payload = results_after_correction[0][0].payload
    print("原始载荷:", payload)
    print("解码载荷:", decoded_payload)
    
    # 检查前9个字节是否匹配（最后一个字节包含CRC）
    payload_match = np.array_equal(payload[:9], decoded_payload[:9])
    print("载荷匹配:", payload_match)
else:
    print("频率校正后未能解码，可能需要调整参数")
