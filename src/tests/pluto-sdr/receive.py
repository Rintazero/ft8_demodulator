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

sample_rate = 1e6 # Hz
center_freq = 2.4e9 - 1e6 # Hz
num_samps = int(sample_rate * 0.16 * 10) # number of samples returned per call to rx()

sdr = adi.Pluto('ip:192.168.2.1')
# sdr.gain_control_mode_chan0 = 'manual'
# sdr.rx_hardwaregain_chan0 = 0.0 # dB
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
sdr.rx_buffer_size = num_samps

samples = sdr.rx() # receive samples off Pluto

# 移除直流偏置
samples = samples - np.mean(samples)

# 设置spectrogram参数
samples_per_symbol = int(sample_rate * 0.16)  # nperseg参数
overlap_samples = samples_per_symbol // 2  # noverlap参数
dft_length = samples_per_symbol  # nfft参数

# 计算频谱图
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

# 绘制频谱图
plt.figure()
plt.pcolormesh(times, frequencies/1e6, 10 * np.log10(np.abs(Sxx)))
plt.colorbar(label='功率谱密度 (dB)')
plt.ylabel('频率偏移 (MHz)')
plt.xlabel('时间 (秒)')
plt.title(f'频谱图 (中心频率: {center_freq/1e6:.1f} MHz)')
plt.tight_layout()  # 自动调整布局
plt.savefig('spectrogram.png', dpi='figure', bbox_inches='tight')
plt.close()





