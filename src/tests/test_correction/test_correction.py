import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import scipy.signal as sci

# 添加正确的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

# 导入本地模块
from src.ft8_tools.ft8_generator.modulator import ft8_generator, ft8_baseband_generator
from src.ft8_tools.ft8_beacon_receiver.frequency_correction import correct_frequency_drift
from src.ft8_tools.ft8_demodulator.spectrogram_analyse import (
    calculate_spectrogram,
    select_frequency_band,
)
from src.ft8_tools.ft8_demodulator.ft8_decode import (
    decode_ft8_message,
    FT8Message,
    FT8DecodeStatus,
)
from src.ft8_tools.ft8_demodulator.ftx_types import FT8Waterfall

def save_spectrogram(spectrogram: np.ndarray, filename: str, title: str) -> None:
    """Save spectrogram to file"""
    try:
        plt.figure(figsize=(10, 6))
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        plt.colorbar(label='Power (dB)')
        plt.title(title)
        plt.xlabel('Time (samples)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Warning: Error saving spectrogram: {e}")
        print("Continuing test...")

def verify_decode_results(results: list) -> None:
    """Verify decode results"""
    assert isinstance(results, list), "Decode results should be a list"
    
    if results:
        for message, status, time_sec, freq_hz, score in results:
            assert isinstance(message, FT8Message), "Decoded message should be FT8Message type"
            assert isinstance(status, FT8DecodeStatus), "Decode status should be FT8DecodeStatus type"
            assert len(message.payload) == 10, "FT8 message payload should be 10 bytes"
            
            print("Decoded message:", message.payload.hex())
            print("CRC check:", status.crc_calculated)
            print("LDPC errors:", status.ldpc_errors)
    else:
        print("Failed to decode any messages")

def real_to_analytic(real_signal):
    """Convert real signal to analytic signal (single-sideband complex signal)
    
    Uses Hilbert transform to generate single-sideband signal, preserving positive frequencies
    
    Note: This function is no longer used as we now directly generate complex signals with ft8_baseband_generator
    This function is kept for reference only
    """
    return sci.hilbert(real_signal)

def test_frequency_correction():
    """Test frequency correction functionality"""
    print("Starting frequency correction test...")
    
    # 设置基本参数
    fs = (1024)*32  # 采样率
    f0 = 300    # 音频频率
    fc = 500      # 载波频率（基带信号）
    sym_bin = 6.25  # 符号频率间隔
    sym_t = 0.16    # 符号时间长度
    
    
    # 测试载荷
    test_payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x50], dtype=np.uint8)
    
    # 生成测试波形数据 - 实信号
    print("Generating FT8 real signal...")
    wave_data = ft8_generator(
        test_payload, 
        fs=fs, 
        f0=f0, 
        fc=fc
    )
    
    # 添加线性频率漂移
    print("Adding linear frequency drift...")
    nsamples = len(wave_data)
    
    # 设置频率漂移参数
    fShift_t0_Hz = 0.0  # 初始频偏
    fShift_k_Hzpsample = 568.0 / fs  # 频偏变化率
    print(f"Frequency drift rate: {fShift_k_Hzpsample} Hz/sample")
    
    # 直接生成复信号（基带信号）
    print("Directly generating FT8 baseband complex signal...")
    wave_complex = ft8_baseband_generator(test_payload, fs, f0)
    
    # 将基带信号上变频到载波频率
    wave_complex = wave_complex * np.exp(1j * 2 * np.pi * fc * np.arange(len(wave_complex)) / fs)

    # 对wave_complex前后补零，补零长度为信号本身长度
    original_length = len(wave_complex)
    print(f"Original signal length: {original_length}")
    zeros_padding = np.zeros(original_length, dtype=complex)
    wave_complex_padded = np.concatenate((zeros_padding, wave_complex, zeros_padding))
    print(f"Padded signal length: {len(wave_complex_padded)}")
    
    # 更新采样点数量以匹配新的信号长度
    nsamples = len(wave_complex_padded)
    
    # 计算复信号的频谱图
    print("Calculating complex signal spectrogram...")
    complex_spectrogram, f_complex, t_complex = calculate_spectrogram(wave_complex_padded, fs, 2, 2)
    save_spectrogram(complex_spectrogram, 'complex_signal_spectrogram.png', 'Complex Signal Spectrogram (with Zero Padding)')
    
    # 应用频率漂移
    t = np.arange(nsamples)
    shift_carrier = np.exp(2j * np.pi * fShift_t0_Hz * t / fs + 
                          2j * np.pi * fShift_k_Hzpsample * t**2 / (2 * fs))
    wave_shifted = wave_complex_padded * shift_carrier
    
    # 绘制偏移载波的频谱图
    print("Calculating shifted carrier spectrogram...")
    shifted_spectrogram, f, t = calculate_spectrogram(np.real(wave_shifted), fs, 2, 2)
    
    # 只保留正频率部分
    positive_freq_mask = f >= 0
    shifted_spectrogram_positive = shifted_spectrogram[positive_freq_mask]
    f_positive = f[positive_freq_mask]
    
    # 保存频谱图
    save_spectrogram(shifted_spectrogram_positive, 'shifted_spectrogram.png', 'Signal Spectrogram After Adding Frequency Drift')

    # 添加高斯噪声
    print("Adding Gaussian noise...")
    # 界限在 20 - 30 左右
    Es_N0_dB = 40  # 信号比上噪声频谱密度(dB)
    
    # 计算信号能量
    signal_energy = np.sum(np.abs(wave_shifted)**2) / len(wave_shifted)
    
    # 根据Es/N0计算噪声功率谱密度
    N0 = signal_energy / (10**(Es_N0_dB/10))
    
    # 计算噪声功率，考虑采样率影响
    # 在复信号中，噪声功率为 N0 * fs / 2
    noise_power = N0 * fs
    print(f"Signal energy: {signal_energy}, Noise power spectral density (N0): {N0}, Es/N0: {Es_N0_dB} dB")
    print(f"Sampling rate: {fs} Hz, Calculated noise power: {noise_power}")
    
    # 生成复高斯噪声
    noise_std = np.sqrt(noise_power/2)  # 复噪声的实部和虚部标准差
    noise_real = np.random.normal(0, noise_std, nsamples)
    noise_imag = np.random.normal(0, noise_std, nsamples)
    noise_complex = noise_real + 1j * noise_imag
    
    # 添加噪声到信号
    wave_noisy = wave_shifted + noise_complex
    
    # 计算带噪声信号的频谱图 - 使用实部进行频谱图计算
    print("Calculating noisy signal spectrogram...")
    orig_spectrogram, f, t = calculate_spectrogram(wave_noisy, fs, 2, 2)
    
    # 只保留正频率部分（单边带）
    positive_freq_mask = f >= 0
    orig_spectrogram_positive = orig_spectrogram[positive_freq_mask]
    f_positive = f[positive_freq_mask]
    
    print("orig_spectrogram_positive:", orig_spectrogram_positive.shape)
    save_spectrogram(orig_spectrogram_positive, 'original_noisy_spectrogram.png', 'Original Noisy Signal with Frequency Drift')
    
    # 创建FT8Waterfall对象
    waterfall = FT8Waterfall(
        mag=orig_spectrogram_positive,
        time_osr=2,
        freq_osr=2
    )
    
    # 应用频率校正 - 直接使用复信号进行校正
    print("Applying frequency correction...")
    wave_corrected, estimated_drift_rate = correct_frequency_drift(
        wave_noisy, 
        fs, 
        sym_bin, 
        sym_t,
        params={
            'nsync_sym': 7,
            'ndata_sym': 58,
            'zscore_threshold': 5,
            'max_iteration_num': 400000,
            'bins_per_tone': 2,
            'steps_per_symbol': 2
        }
    )
    
    print(f"Estimated frequency drift rate: {estimated_drift_rate} Hz/sample")
    print(f"Actual frequency drift rate: {fShift_k_Hzpsample} Hz/sample")
    print(f"Frequency drift estimation error: {(estimated_drift_rate - fShift_k_Hzpsample) * nsamples} Hz")
    
    # 计算校正后信号的频谱图
    print("Calculating corrected signal spectrogram...")
    corrected_spectrogram, f, t = calculate_spectrogram(np.real(wave_corrected), fs, 2, 2)
    
    # 只保留正频率部分
    positive_freq_mask = f >= 0
    corrected_spectrogram_positive = corrected_spectrogram[positive_freq_mask]
    f_positive = f[positive_freq_mask]
    
    save_spectrogram(corrected_spectrogram_positive, 'corrected_spectrogram.png', 'Frequency Corrected Signal Spectrogram (Zero Padded)')
    
    # 解码校正后的信号 - 使用实部进行解码
    print("Decoding corrected signal...")
    results_after_correction = decode_ft8_message(
        wave_data=np.real(wave_corrected),
        sample_rate=fs,
        bins_per_tone=2,
        steps_per_symbol=2,
        max_candidates=100,
        min_score=6,
        max_iterations=40,
        # 限制解码范围，用于加快测试
        freq_min=0,  # 最小频率限制 (Hz)
        freq_max=1500,  # 最大频率限制 (Hz)
        time_min=10,  # 最小时间限制 (秒)
        time_max=None   # 最大时间限制 (秒)
    )
    
    print(f"Number of decode results after frequency correction: {len(results_after_correction)}")
    if results_after_correction:
        print("Successfully decoded after frequency correction!")
        verify_decode_results(results_after_correction)
    else:
        print("Failed to decode after frequency correction, may need to adjust parameters")
    
    # 比较原始载荷和解码结果
    if results_after_correction:
        decoded_payload = results_after_correction[0][0].payload
        print("Original payload:", test_payload)
        print("Decoded payload:", decoded_payload)
        
        # 检查前9个字节是否匹配（最后一个字节包含CRC）
        payload_match = np.array_equal(test_payload[:9], decoded_payload[:9])
        print(f"Payload match: {payload_match}")
    
    print("Frequency correction test completed!")

if __name__ == "__main__":
    print("Starting tests...")
    test_frequency_correction()
    print("All tests completed!")
