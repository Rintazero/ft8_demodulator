import numpy as np
import matplotlib.pyplot as plt
import adi

def plot_fft(samples, sample_rate, center_freq):
    """
    对采样数据进行FFT分析并绘图
    
    参数:
    samples: 采样数据
    sample_rate: 采样率 (Hz)
    center_freq: 中心频率 (Hz)
    """
    # 计算FFT
    psd = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2
    psd_dB = 10*np.log10(psd)
    f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))
    
    # 绘制FFT
    plt.figure(figsize=(12, 6))
    plt.plot(f/1e6, psd_dB)
    plt.grid(True)
    plt.xlabel('频率偏移 (MHz)')
    plt.ylabel('功率谱密度 (dB)')
    plt.title(f'FFT分析 (中心频率: {center_freq/1e6:.1f} MHz)')
    plt.tight_layout()
    plt.show()

def main():
    # SDR参数设置
    sample_rate = 10e6  # 10 MHz
    center_freq = 1099e6  # 1099 MHz
    num_samples = 16384  # 采样点数
    
    try:
        # 初始化SDR
        print("正在初始化SDR...")
        sdr = adi.Pluto('ip:192.168.3.2')
        sdr.gain_control_mode_chan0 = 'slow_attack'
        sdr.rx_hardwaregain_chan0 = 50
        sdr.rx_lo = int(center_freq)
        sdr.sample_rate = int(sample_rate)
        sdr.rx_rf_bandwidth = int(sample_rate * 0.8)
        sdr.rx_buffer_size = num_samples
        
        print("正在接收数据...")
        samples = sdr.rx()
        
        # 去除DC偏置
        samples = samples - np.mean(samples)
        print(f"已接收 {len(samples)} 个采样点")
        
        # 绘制FFT
        plot_fft(samples, sample_rate, center_freq)
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 