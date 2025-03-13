import numpy as np
import wave
import sys
import os
from pathlib import Path
import argparse

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules
from ft8_tools.ft8_demodulator.ft8_decode import decode_ft8_message

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
                        bins_per_tone: int = 2,
                        steps_per_symbol: int = 2,
                        max_candidates: int = 20,
                        min_score: float = 10,
                        max_iterations: int = 20) -> list:
    """
    从wave文件中解码FT8信号
    
    Args:
        wave_path: wave文件路径
        freq_min: 最小频率限制 (Hz)，None表示不限制
        freq_max: 最大频率限制 (Hz)，None表示不限制
        bins_per_tone: 每个音调的频率bin数
        steps_per_symbol: 每个符号的时间步数
        max_candidates: 最大候选数量
        min_score: 最小分数阈值
        max_iterations: 最大迭代次数
    
    Returns:
        list: 解码结果列表
    """
    # 读取wave文件
    wave_data, sample_rate = read_wave_file(wave_path)
    
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
        freq_max=freq_max
    )
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从wave文件解码FT8信号')
    parser.add_argument('wave_file', help='输入的wave文件路径')
    parser.add_argument('--freq-min', type=float, help='最小频率限制 (Hz)')
    parser.add_argument('--freq-max', type=float, help='最大频率限制 (Hz)')
    parser.add_argument('--bins-per-tone', type=int, default=2, help='每个音调的频率bin数')
    parser.add_argument('--steps-per-symbol', type=int, default=2, help='每个符号的时间步数')
    parser.add_argument('--max-candidates', type=int, default=20, help='最大候选数量')
    parser.add_argument('--min-score', type=float, default=10, help='最小分数阈值')
    parser.add_argument('--max-iterations', type=int, default=20, help='最大迭代次数')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.wave_file):
        print(f"Error: File {args.wave_file} does not exist")
        sys.exit(1)
    
    try:
        results = decode_ft8_from_wave(
            args.wave_file,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            bins_per_tone=args.bins_per_tone,
            steps_per_symbol=args.steps_per_symbol,
            max_candidates=args.max_candidates,
            min_score=args.min_score,
            max_iterations=args.max_iterations
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
            
    except Exception as e:
        print(f"Error decoding wave file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
