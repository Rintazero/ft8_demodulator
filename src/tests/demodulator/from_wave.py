import numpy as np
import wave
import sys
import os
from pathlib import Path
import argparse
import scipy.signal

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules
from ft8_tools.ft8_demodulator.ft8_decode import decode_ft8_message
from ft8_tools.ft8_beacon_receiver.frequency_correction import correct_frequency_drift
from ft8_tools.ft8_demodulator.spectrogram_analyse import (
    calculate_spectrogram,
    FT8_SYMBOL_DURATION_S,
    FT8_SYMBOL_FREQ_INTERVAL_HZ
)

from ft8_tools.ft8_demodulator.ft8_decode import create_waterfall_from_spectrogram
from ft8_tools.ft8_demodulator.ftx_types import FT8Waterfall

def read_wave_file(wave_path: str) -> tuple[np.ndarray, int]:
    """
    从wave文件中读取音频数据
    
    Args:
        wave_path: wave文件路径
    
    Returns:
        tuple: (音频数据, 采样率)
    """
    with wave.open(wave_path, 'rb') as wave_file:
        # 获取wave文件的基本参数
        n_channels = wave_file.getnchannels()
        sample_width = wave_file.getsampwidth()
        sample_rate = wave_file.getframerate()
        n_frames = wave_file.getnframes()
        

        print(f"n_channels: {n_channels}")
        print(f"sample_width: {sample_width}")
        print(f"sample_rate: {sample_rate}")
        print(f"n_frames: {n_frames}")
        # 读取音频数据
        wave_data = wave_file.readframes(n_frames)
        
        # 将字节数据转换为numpy数组
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        wave_data = np.frombuffer(wave_data, dtype=dtype)
        
        # 如果是立体声，只取一个声道
        if n_channels == 2:
            wave_data = wave_data[::2]
        
        # 将数据归一化到[-1, 1]范围
        wave_data = wave_data.astype(np.float32)
        wave_data /= np.iinfo(dtype).max
        
        return wave_data, sample_rate

def decode_ft8_from_wave(wave_path: str, 
                        freq_min: float = None,  # 最小频率限制 (Hz)
                        freq_max: float = None,  # 最大频率限制 (Hz)
                        time_min: float = None,  # 最小时间限制 (秒)
                        time_max: float = None,  # 最大时间限制 (秒)
                        bins_per_tone: int = 2,
                        steps_per_symbol: int = 2,
                        max_candidates: int = 20,
                        min_score: float = 10,
                        max_iterations: int = 20,
                        correction: bool = False) -> list:
    """
    从wave文件中解码FT8信号
    
    Args:
        wave_path: wave文件路径
        freq_min: 最小频率限制 (Hz)，None表示不限制
        freq_max: 最大频率限制 (Hz)，None表示不限制
        time_min: 最小时间限制 (秒)，None表示不限制
        time_max: 最大时间限制 (秒)，None表示不限制
        bins_per_tone: 每个音调的频率bin数
        steps_per_symbol: 每个符号的时间步数
        max_candidates: 最大候选数量
        min_score: 最小分数阈值
        max_iterations: 最大迭代次数
        correction: 是否进行频偏矫正
    
    Returns:
        list: 解码结果列表
    """
    # 读取wave文件
    wave_data, sample_rate = read_wave_file(wave_path)
    # wave_data = wave_data + 1j * scipy.signal.hilbert(wave_data)

    if correction:
        print("正在进行频偏矫正...")
        # 计算频谱图用于频偏矫正
        spectrogram, f, t = calculate_spectrogram(
            wave_data, sample_rate, bins_per_tone, steps_per_symbol
        )
        
        # 只取正频率部分
        positive_freq_mask = f >= 0
        spectrogram = spectrogram[positive_freq_mask]
        f = f[positive_freq_mask]
        
        # 应用频率限制
        if freq_min is not None or freq_max is not None:
            freq_min = freq_min if freq_min is not None else f[0]
            freq_max = freq_max if freq_max is not None else f[-1]
            freq_mask = (f >= freq_min) & (f <= freq_max)
            spectrogram = spectrogram[freq_mask]
            f = f[freq_mask]
        
        # 应用时间限制
        if time_min is not None or time_max is not None:
            time_min = time_min if time_min is not None else t[0]
            time_max = time_max if time_max is not None else t[-1]
            time_mask = (t >= time_min) & (t <= time_max)
            spectrogram = spectrogram[:, time_mask]
            t = t[time_mask]
        
        spectrogram

        import matplotlib.pyplot as plt
        # 绘制频谱图
        plt.figure(figsize=(10, 6))
        plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
        plt.colorbar(label='强度 (dB)')
        plt.title('FT8信号频谱图')
        plt.xlabel('时间 (秒)')
        plt.ylabel('频率 (Hz)')
        plt.savefig('ft8_spectrogram_correction.png')
        plt.close()

        # 创建瀑布图数据结构
        waterfall = create_waterfall_from_spectrogram(
            spectrogram, steps_per_symbol, bins_per_tone
        )
        
        # 进行频偏矫正
        wave_data, drift_rate = correct_frequency_drift(
            wave_data + 1j * scipy.signal.hilbert(wave_data),  # 转换为解析信号
            sample_rate,
            FT8_SYMBOL_FREQ_INTERVAL_HZ,
            FT8_SYMBOL_DURATION_S,
            waterfall
        )
        print(f"估计的频率漂移率: {drift_rate * sample_rate:.2f} Hz/s")
        

    
    # 解码FT8消息
    results = decode_ft8_message(
        wave_data=wave_data,
        sample_rate=sample_rate,
        bins_per_tone=bins_per_tone,
        steps_per_symbol=steps_per_symbol,
        max_candidates=max_candidates,
        min_score=min_score,
        max_iterations=max_iterations,
        freq_min=freq_min,
        freq_max=freq_max,
        time_min=time_min,
        time_max=time_max
    )
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从wave文件解码FT8信号')
    parser.add_argument('wave_file', help='输入的wave文件路径')
    parser.add_argument('--freq-min', type=float, help='最小频率限制 (Hz)')
    parser.add_argument('--freq-max', type=float, help='最大频率限制 (Hz)')
    parser.add_argument('--time-min', type=float, help='最小时间限制 (秒)')
    parser.add_argument('--time-max', type=float, help='最大时间限制 (秒)')
    parser.add_argument('--bins-per-tone', type=int, default=2, help='每个音调的频率bin数')
    parser.add_argument('--steps-per-symbol', type=int, default=2, help='每个符号的时间步数')
    parser.add_argument('--max-candidates', type=int, default=20, help='最大候选数量')
    parser.add_argument('--min-score', type=float, default=10, help='最小分数阈值')
    parser.add_argument('--max-iterations', type=int, default=20, help='最大迭代次数')
    parser.add_argument('--correction', type=bool, default=False, help='是否进行频偏矫正')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.wave_file):
        print(f"Error: File {args.wave_file} does not exist")
        sys.exit(1)
    

    results = decode_ft8_from_wave(
        args.wave_file,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        time_min=args.time_min,
        time_max=args.time_max,
        bins_per_tone=args.bins_per_tone,
        steps_per_symbol=args.steps_per_symbol,
        max_candidates=args.max_candidates,
        min_score=args.min_score,
        max_iterations=args.max_iterations,
        correction=args.correction
    )
    
    if not results:
        print("No FT8 messages decoded")
        return
    
    print("\nDecoded FT8 messages:")
    print("-" * 50)
    for message, status, time_sec, freq_hz, score in results:
        print(f"Time: {time_sec:.2f} seconds")
        print(f"Frequency: {freq_hz:.1f} Hz")
        print(f"Score: {score:.1f}")
        print(f"Payload: {message.payload.hex()}")
        print(f"CRC check: {status.crc_calculated}")
        print(f"LDPC errors: {status.ldpc_errors}")
        print("-" * 50)
            


if __name__ == "__main__":
    main()
