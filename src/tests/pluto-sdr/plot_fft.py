import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import adi

def plot_fft(samples, sample_rate, center_freq, save_path='fft_analysis.png'):
    """
    对采样数据进行FFT分析并绘图
    
    参数:
    samples: 采样数据
    sample_rate: 采样率 (Hz)
    center_freq: 中心频率 (Hz)
    save_path: 保存图片的路径
    """
    # Calculate FFT
    fft_size = len(samples)
    window = np.blackman(fft_size)  # 使用Blackman窗口减少频谱泄漏
    fft_result = np.fft.fft(samples * window)
    fft_result = np.fft.fftshift(fft_result)
    fft_freq = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/sample_rate))

    # Plot FFT
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freq/1e6, 20 * np.log10(np.abs(fft_result)))
    plt.grid(True)
    plt.xlabel('Frequency Offset (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'FFT Analysis (Center Frequency: {center_freq/1e6:.1f} MHz)')
    plt.xlim(-sample_rate/2e6, sample_rate/2e6)  # Set frequency limits to ±fs/2
    plt.ylim(20, 120)  # 限制幅度范围
    plt.tight_layout()
    plt.savefig(save_path, dpi='figure', bbox_inches='tight')
    plt.close()

def receive_from_sdr():
    """从SDR接收数据"""
    # SDR参数
    sample_rate = 10e6  # Hz
    center_freq = 1000e6  # Hz
    samples_per_buffer = int(sample_rate * 0.16)  # 一个符号周期的样本数

    # 初始化SDR
    print("Initializing SDR...")
    sdr = adi.Pluto('ip:192.168.3.2')
    sdr.gain_control_mode_chan0 = 'manual'  # 使用AGC
    sdr.rx_hardwaregain_chan0 = 0  # 降低增益到50dB
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(sample_rate * 0.8)  # 减小带宽以降低噪声
    sdr.rx_buffer_size = samples_per_buffer

    # 接收数据
    print("Receiving samples...")
    samples = sdr.rx()
    
    # 移除DC偏置
    samples = samples - np.mean(samples)
    print(f"Received {len(samples)} samples")
    
    return samples, sample_rate, center_freq

def continuous_fft_plot():
    """持续接收数据并绘制FFT"""
    try:
        while True:
            # 接收数据
            samples, sample_rate, center_freq = receive_from_sdr()
            
            # 绘制FFT
            plot_fft(samples, sample_rate, center_freq)
            print("FFT plot updated")
            
            # 等待用户输入
            user_input = input("Press Enter to update FFT plot, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    continuous_fft_plot() 