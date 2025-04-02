import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import FT8 demodulation modules
from ft8_tools.ft8_demodulator.ft8_decode import decode_ft8_message

# SDR parameters
sample_rate = 1e6  # Hz
center_freq = 1000e6  # Hz
samples_per_buffer = int(sample_rate * 0.16)  # 一个符号周期的样本数
num_buffers = 30  # 需要采集的缓冲区数量

# Initialize SDR
sdr = adi.Pluto('ip:192.168.3.2')
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = -20 # dB
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
sdr.rx_buffer_size = samples_per_buffer

# 采集多个缓冲区的数据
print("Collecting samples...")
samples = np.array([])
for i in range(num_buffers):
    buffer_samples = sdr.rx()
    samples = np.concatenate([samples, buffer_samples])
    print(f"Collected buffer {i+1}/{num_buffers}")

# Remove DC offset
samples = samples - np.mean(samples)
print(f"Total samples collected: {len(samples)}")
print(f"First 10 samples: {samples[:10]}")

# Spectrogram parameters
samples_per_symbol = int(sample_rate * 0.16)  # nperseg parameter
overlap_samples = samples_per_symbol // 2  # noverlap parameter
dft_length = samples_per_symbol  # nfft parameter

# Calculate spectrogram
frequencies, times, Sxx = signal.spectrogram(
    samples,
    fs=sample_rate,
    window='hann',
    nperseg=samples_per_symbol,
    noverlap=overlap_samples,
    nfft=dft_length,
    detrend=False,
    return_onesided=False,
    scaling='spectrum'
)

# Shift zero frequency to center
frequencies = np.fft.fftshift(frequencies)
Sxx = np.fft.fftshift(Sxx, axes=0)

# Decode the signal
print("Attempting to decode FT8 message...")
decode_results = decode_ft8_message(
    wave_data=samples,
    sample_rate=sample_rate,
    bins_per_tone=2,
    steps_per_symbol=2,
    max_candidates=100,
    min_score=5,
    max_iterations=20,
    # freq_min=-1000,  # 设置最小频率为 -3kHz
    freq_max=10000    # 设置最大频率为 +3kHz
)

# Print decode results
if decode_results:
    print("\nSuccessfully decoded messages:")
    for message, status, time_sec, freq_hz, score in decode_results:
        print(f"Message payload: {message.payload.hex()}")
        print(f"CRC check: {status.crc_calculated}")
        print(f"LDPC errors: {status.ldpc_errors}")
        print(f"Time: {time_sec:.2f} s")
        print(f"Frequency: {freq_hz:.1f} Hz")
        print(f"Score: {score}\n")
else:
    print("No messages were successfully decoded.")

# Save samples data
samples_file = 'received_samples.npy'
np.save(samples_file, samples)
print(f"Saved samples to {samples_file}")

# Plot spectrogram
plt.figure(figsize=(12, 6))
plt.imshow(10 * np.log10(np.abs(Sxx)), 
          aspect='auto', 
          origin='lower',
          extent=[times[0], times[-1], frequencies[0]/1e6, frequencies[-1]/1e6])
plt.colorbar(label='Power Spectral Density (dB)')
plt.ylabel('Frequency Offset (MHz)')
plt.xlabel('Time (s)')
plt.title(f'Spectrogram (Center Frequency: {center_freq/1e6:.1f} MHz)')
plt.tight_layout()
plt.savefig('spectrogram.png', dpi='figure', bbox_inches='tight')
plt.close()





