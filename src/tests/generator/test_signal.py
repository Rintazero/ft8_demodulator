import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ft8_tools import ft8_generator

payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)

fs = 2e3
f0 = 0
fc = 500

ft8_signal_baseband = ft8_generator.ft8_baseband_generator(payload, fs, f0)

ft8_signal = ft8_generator.ft8_generator(payload, fs, f0, fc)

# Define low-pass filter
def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Apply the low-pass filter to rx_signal
rx_signal = lowpass_filter(ft8_signal * np.exp(-1j*2*np.pi*fc*np.arange(len(ft8_signal))/fs), cutoff=100, fs=fs)

# 绘制 ft8_signal_baseband 的时频图
plt.figure(figsize=(10, 6))
plt.specgram(ft8_signal_baseband, NFFT=256, Fs=fs, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.title('FT8 Signal Baseband Time-Frequency Representation')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude (dB)')
plt.grid()
plt.show()
plt.savefig('ft8_signal_baseband.png')

# 绘制 ft8_signal 的时频图
plt.figure(figsize=(10, 6))
plt.specgram(ft8_signal, NFFT=256, Fs=fs, Fc=0, noverlap=128, cmap='viridis', sides='onesided', mode='default')
plt.title('FT8 Signal Time-Frequency Representation')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude (dB)')
plt.grid()
plt.show()
plt.savefig('ft8_signal.png')
# 绘制 rx_signal 的时频图
plt.figure(figsize=(10, 6))
plt.specgram(rx_signal, NFFT=256, Fs=fs, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.title('FT8 Signal Time-Frequency Representation')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude (dB)')
plt.grid()
plt.show()
plt.savefig('rx_signal.png')




