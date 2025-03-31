import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import os
import wave

fs = 1e3
f0 = 300
f1 = 100
f2 = 50

t_s = 5

num_samples = int(t_s * fs)

# 生成基带信号
payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
baseband_signal = ft8_baseband_generator(payload, fs, f0)

# 生成载波
signal = baseband_signal * np.exp(1j * 2 * np.pi * f0 * np.arange(num_samples) / fs)

# 保存复数信号
np.save(os.path.join(save_path, 'signal.npy'), signal)

# 保存实部作为WAV文件
wavfile.write(os.path.join(save_path, 'signal.wav'), int(fs), np.real(signal).astype(np.float32))

rx_signal = signal * np.exp(-1j * 2 * np.pi * f0 * np.arange(num_samples) / fs)

plt.figure(figsize=(10, 6))
plt.specgram(rx_signal, NFFT=256, Fs=fs, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.colorbar(label='Intensity (dB)')
plt.title('Signal Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')  # Changed from Hz to kHz
plt.show()

# python ./tests/test_signal.py

