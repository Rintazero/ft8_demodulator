import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

fs = 1e3
f0 = 300
f1 = 100
f2 = 50

t_s = 5

num_samples = int(t_s * fs)

baseband_I = np.sin(2 * np.pi * f1 * np.arange(num_samples/2) / fs )
baseband_Q = -1 * np.cos(2 * np.pi * f1 * np.arange(num_samples/2) / fs )

signal = (baseband_I + 1j * baseband_Q) * np.exp(1j * 2 * np.pi * f0 * np.arange(num_samples) / fs)

rx_signal = signal * np.exp(-1j * 2 * np.pi * f0 * np.arange(num_samples) / fs)

plt.figure(figsize=(10, 6))
plt.specgram(rx_signal, NFFT=256, Fs=fs, Fc=0, noverlap=128, cmap='viridis', sides='twosided', mode='default')
plt.colorbar(label='Intensity (dB)')
plt.title('Signal Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')  # Changed from Hz to kHz
plt.show()

# python ./tests/test_signal.py

