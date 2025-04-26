import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from pathlib import Path
import scipy.signal as sci

# 添加正确的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

# 配置日志记录器，只设置当前模块的logger为DEBUG级别
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 如果还没有处理器，添加一个控制台处理器
if not logger.handlers:
    print("Adding console handler")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

# 设置matplotlib使用西文字体，避免中文字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用DejaVu Sans字体
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

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

def save_spectrogram(spectrogram: np.ndarray, filename: str, title: str, debug_plot: bool = True) -> None:
    """如果debug_plot为True，保存频谱图到文件"""
    if not debug_plot:
        return
        
    try:
        plt.figure(figsize=(10, 6))
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        plt.colorbar(label='Power (dB)')
        # 使用英文标题避免中文字体问题
        plt.title(title)
        plt.xlabel('Time (samples)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        logger.warning("保存频谱图时出错: %s", e)
        logger.warning("继续测试...")

def verify_decode_results(results: list) -> None:
    """验证解码结果"""
    assert isinstance(results, list), "解码结果应该是列表类型"
    
    if results:
        for message, status, time_sec, freq_hz, score in results:
            assert isinstance(message, FT8Message), "解码消息应为FT8Message类型"
            assert isinstance(status, FT8DecodeStatus), "解码状态应为FT8DecodeStatus类型"
            assert len(message.payload) == 10, "FT8消息载荷应为10字节"
            
            logger.debug("解码消息: %s", message.payload.hex())
            logger.debug("CRC校验: %s", status.crc_calculated)
            logger.debug("LDPC错误: %s", status.ldpc_errors)
    else:
        logger.debug("未能解码任何消息")

def real_to_analytic(real_signal):
    """将实信号转换为解析信号（单边带复信号）
    
    使用希尔伯特变换生成单边带信号，保留正频率
    
    注意：此函数不再使用，因为我们现在直接使用ft8_baseband_generator生成复信号
    保留此函数仅供参考
    """
    return sci.hilbert(real_signal)

def test_frequency_correction(params=None, debug_plot=True):
    """测试频率校正功能
    
    参数:
    params: 可选参数字典，包含以下字段:
        - fs: 采样率 (默认: 32768 Hz)
        - f0: 音频频率 (默认: 300 Hz)
        - fc: 载波频率 (默认: 500 Hz)
        - sym_bin: 符号频率间隔 (默认: 6.25 Hz)
        - sym_t: 符号时间长度 (默认: 0.16 s)
        - payload: 测试载荷 (默认: [0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x50])
        - fShift_t0_Hz: 初始频偏 (默认: 0.0 Hz)
        - fShift_k_Hz: 频偏变化率 (默认: 568.0 Hz/s)
        - Es_N0_dB: 信噪比 (默认: 40 dB)
        - time_min: 解码最小时间限制 (默认: 10 s)
        - time_max: 解码最大时间限制 (默认: None)
        - correction_params: 频率校正参数字典 (详见 correct_frequency_drift 函数)
    debug_plot: 是否生成调试图像 (默认: True)
    
    返回:
    tuple: (校正后的信号, 估计的频率漂移率, 实际的频率漂移率, 解码结果)
    """
    logger.debug("开始频率校正测试...")
    
    # 设置默认参数
    default_params = {
        'fs': (1024)*32,  # 采样率
        'f0': 300,        # 音频频率
        'fc': 500,        # 载波频率（基带信号）
        'sym_bin': 6.25,  # 符号频率间隔
        'sym_t': 0.16,    # 符号时间长度
        'payload': np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x50], dtype=np.uint8),
        'fShift_t0_Hz': 0.0,  # 初始频偏
        'fShift_k_Hz': 568.0,  # 频偏变化率(Hz/s)
        'Es_N0_dB': 25,   # 信号比上噪声频谱密度(dB)
        'time_min': 10,   # 最小时间限制 (秒)
        'time_max': None, # 最大时间限制 (秒)
        'correction_params': {
            'nsync_sym': 7,
            'ndata_sym': 58,
            'zscore_threshold': 5,
            'max_iteration_num': 400000,
            'bins_per_tone': 2,
            'steps_per_symbol': 8,
            # 'window_size_factor': 8,  # 窗口大小因子，用于计算window_size
            # 'max_variance_factor': 0.01  # 方差因子，实际方差阈值将乘以频谱点数的平方
        }
    }
    
    if params is None:
        params = default_params
    else:
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
    
    # 解包参数
    fs = params['fs']
    f0 = params['f0']
    fc = params['fc']
    sym_bin = params['sym_bin']
    sym_t = params['sym_t']
    test_payload = params['payload']
    fShift_t0_Hz = params['fShift_t0_Hz']
    fShift_k_Hz = params['fShift_k_Hz']
    Es_N0_dB = params['Es_N0_dB']
    time_min = params['time_min']
    time_max = params['time_max']
    correction_params = params['correction_params']
    
    # 生成测试波形数据 - 实信号
    logger.debug("生成FT8实信号...")
    wave_data = ft8_generator(
        test_payload, 
        fs=fs, 
        f0=f0, 
        fc=fc
    )
    
    # 添加线性频率漂移
    logger.debug("添加线性频率漂移...")
    
    # 直接生成复信号（基带信号）
    logger.debug("直接生成FT8基带复信号...")
    wave_complex = ft8_baseband_generator(test_payload, fs, f0)
    
    # 将基带信号上变频到载波频率
    wave_complex = wave_complex * np.exp(1j * 2 * np.pi * fc * np.arange(len(wave_complex)) / fs)

    # 对wave_complex前后补零，补零长度为信号本身长度
    original_length = len(wave_complex)
    logger.debug("原始信号长度: %d", original_length)
    zeros_padding = np.zeros(original_length, dtype=complex)
    wave_complex_padded = np.concatenate((zeros_padding, wave_complex, zeros_padding))
    logger.debug("补零后信号长度: %d", len(wave_complex_padded))
    
    # 更新采样点数量以匹配新的信号长度
    nsamples = len(wave_complex_padded)
    
    # 计算复信号的频谱图
    logger.debug("计算复信号频谱图...")
    complex_spectrogram, f_complex, t_complex = calculate_spectrogram(wave_complex_padded, fs, 2, 2)
    save_spectrogram(complex_spectrogram, 'complex_signal_spectrogram.png', 
                     'Complex Signal Spectrogram (with Zero Padding)', debug_plot)
    
    # 计算每个采样点的频偏变化率
    fShift_k_Hzpsample = fShift_k_Hz / fs
    logger.debug("频率漂移率: %f Hz/sample", fShift_k_Hzpsample)
    
    # 应用频率漂移
    t = np.arange(nsamples)
    shift_carrier = np.exp(2j * np.pi * fShift_t0_Hz * t / fs + 
                          2j * np.pi * fShift_k_Hzpsample * t**2 / (2 * fs))
    wave_shifted = wave_complex_padded * shift_carrier
    
    # 计算频移后载波的频谱图
    logger.debug("计算频移后的载波频谱图...")
    shifted_spectrogram, f, t = calculate_spectrogram(np.real(wave_shifted), fs, 2, 2)
    
    # 只保留正频率部分
    positive_freq_mask = f >= 0
    shifted_spectrogram_positive = shifted_spectrogram[positive_freq_mask]
    f_positive = f[positive_freq_mask]
    
    # 保存频谱图
    save_spectrogram(shifted_spectrogram_positive, 'shifted_spectrogram.png', 
                     'Signal Spectrogram After Adding Frequency Drift', debug_plot)

    # 添加高斯噪声
    logger.debug("添加高斯噪声...")
    
    # 计算信号能量
    signal_energy = np.sum(np.abs(wave_shifted)**2) / len(wave_shifted)
    
    # 根据Es/N0计算噪声功率谱密度
    N0 = signal_energy / (10**(Es_N0_dB/10))
    
    # 计算噪声功率，考虑采样率影响
    # 在复信号中，噪声功率为 N0 * fs / 2
    noise_power = N0 * fs
    logger.debug("信号能量: %f, 噪声功率谱密度 (N0): %f, Es/N0: %d dB", signal_energy, N0, Es_N0_dB)
    logger.debug("采样率: %d Hz, 计算的噪声功率: %f", fs, noise_power)
    
    # 生成复高斯噪声
    noise_std = np.sqrt(noise_power/2)  # 复噪声的实部和虚部标准差
    noise_real = np.random.normal(0, noise_std, nsamples)
    noise_imag = np.random.normal(0, noise_std, nsamples)
    noise_complex = noise_real + 1j * noise_imag
    
    # 添加噪声到信号
    wave_noisy = wave_shifted + noise_complex
    
    # 计算带噪声信号的频谱图 - 使用实部进行频谱图计算
    logger.debug("计算带噪声信号的频谱图...")
    orig_spectrogram, f, t = calculate_spectrogram(wave_noisy, fs, 2, 2)
    
    # 只保留正频率部分（单边带）
    positive_freq_mask = f >= 0
    orig_spectrogram_positive = orig_spectrogram[positive_freq_mask]
    f_positive = f[positive_freq_mask]
    
    logger.debug("orig_spectrogram_positive形状: %s", orig_spectrogram_positive.shape)
    save_spectrogram(orig_spectrogram_positive, 'original_noisy_spectrogram.png', 
                     'Original Noisy Signal with Frequency Drift', debug_plot)
    
    # 创建FT8Waterfall对象
    waterfall = FT8Waterfall(
        mag=orig_spectrogram_positive,
        time_osr=2,
        freq_osr=2
    )
    
    # 应用频率校正 - 直接使用复信号进行校正
    logger.debug("应用频率校正...")
    wave_corrected, estimated_drift_rate = correct_frequency_drift(
        wave_noisy, 
        fs, 
        sym_bin, 
        sym_t,
        params=correction_params
    )
    
    logger.debug("估计的频率漂移率: %f Hz/sample", estimated_drift_rate)
    logger.debug("实际的频率漂移率: %f Hz/sample", fShift_k_Hzpsample)
    logger.debug("频率漂移估计误差: %f Hz", (estimated_drift_rate - fShift_k_Hzpsample) * nsamples)
    
    # 计算校正后信号的频谱图
    logger.debug("计算校正后信号的频谱图...")
    corrected_spectrogram, f, t = calculate_spectrogram(np.real(wave_corrected), fs, 2, 2)
    
    # 只保留正频率部分
    positive_freq_mask = f >= 0
    corrected_spectrogram_positive = corrected_spectrogram[positive_freq_mask]
    f_positive = f[positive_freq_mask]
    
    # 绘制校正后的频谱图
    if debug_plot:
        plt.figure(figsize=(12, 6))
        plt.imshow(corrected_spectrogram_positive, aspect='auto', origin='lower', 
                   extent=[t[0], t[-1], f_positive[0], f_positive[-1]])
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Corrected Signal Spectrogram')
        plt.grid(True)
        plt.savefig('corrected_spectrogram_display.png')
        plt.close()
    
    save_spectrogram(corrected_spectrogram_positive, 'corrected_spectrogram.png', 
                     'Frequency Corrected Signal Spectrogram (Zero Padded)', debug_plot)
    
    # 解码校正后的信号 - 使用实部进行解码
    logger.debug("解码校正后的信号...")
    results_after_correction = decode_ft8_message(
        wave_data=np.real(wave_corrected),
        sample_rate=fs,
        bins_per_tone=2,
        steps_per_symbol=2,
        max_candidates=100,
        min_score=6,
        max_iterations=40,
        # 限制解码范围，用于加快测试
        # freq_min=0,  # 最小频率限制 (Hz)
        # freq_max=1500,  # 最大频率限制 (Hz)
        time_min=time_min,  # 最小时间限制 (秒)
        time_max=time_max   # 最大时间限制 (秒)
    )
    
    logger.debug("频率校正后的解码结果数量: %d", len(results_after_correction))
    if results_after_correction:
        logger.debug("频率校正后成功解码!")
        verify_decode_results(results_after_correction)
    else:
        logger.debug("频率校正后未能解码，可能需要调整参数")
    
    # 比较原始载荷和解码结果
    payload_match = False
    if results_after_correction:
        decoded_payload = results_after_correction[0][0].payload
        logger.debug("原始载荷: %s", test_payload)
        logger.debug("解码载荷: %s", decoded_payload)
        
        # 检查前9个字节是否匹配（最后一个字节包含CRC）
        payload_match = np.array_equal(test_payload[:9], decoded_payload[:9])
        logger.debug("载荷匹配: %s", payload_match)
    
    logger.debug("频率校正测试完成!")
    
    # 返回结果
    return wave_corrected, estimated_drift_rate, fShift_k_Hzpsample, results_after_correction

if __name__ == "__main__":
    logger.debug("开始测试...")
    test_frequency_correction(debug_plot=True)
    logger.debug("所有测试完成!")
