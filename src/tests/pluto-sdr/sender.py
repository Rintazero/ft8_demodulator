import numpy as np
import adi
import sys
import os
import time

# 添加 src 目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入 ft8 生成器
from ft8_tools.ft8_generator import ft8_generator

# SDR 参数设置
sample_rate = 1e6  # Hz，与 FT8 采样率保持一致
center_freq = 1000e6  # Hz

# 连接到 PlutoSDR
sdr = adi.Pluto("ip:192.168.3.2")
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -50  # 发送功率，范围 -90 到 0 dB

# FT8 参数
fs = sample_rate  # 使用相同的采样率
f0 = 500    # 音频频率
fc = 0      # 载波频率
test_payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)

# 生成 FT8 信号
ft8_samples = ft8_generator(
    test_payload,
    fs=fs,
    f0=f0,
    fc=fc
)

print("生成FT8信号")

# 调整信号幅度
ft8_samples_scaled = ft8_samples * (2**14)

print(ft8_samples[:10])

# 一次性发送整个信号
while True:
    print("发送FT8信号")
    sdr.tx(ft8_samples_scaled)
    time.sleep(15)

print("FT8 信号发送完成！")