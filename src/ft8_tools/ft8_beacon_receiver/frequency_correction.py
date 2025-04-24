import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from ..ft8_demodulator.ftx_types import FT8Waterfall
from sklearn.preprocessing import PolynomialFeatures
from ..ft8_demodulator.spectrogram_analyse import calculate_spectrogram
from ..ft8_demodulator.ft8_decode import create_waterfall_from_spectrogram

# 设置matplotlib字体设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用DejaVu Sans以获得更好的兼容性
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

def gfsk_pulse(bt, t):
    """
    生成GFSK脉冲
    
    参数:
    bt: 带宽时间乘积
    t: 时间序列
    
    返回:
    output: GFSK脉冲
    """
    k = np.pi * np.sqrt(2.0/np.log(2.0))
    output = 0.5 * (scipy.special.erf(k*bt*(t+0.5)) - scipy.special.erf(k*bt*(t-0.5)))
    return output

def detect_signal_continuity(max_freq_indices, window_size=8, max_variance=10.0):
    """
    通过分析最大频率索引的连续性来检测信号
    
    参数:
    max_freq_indices: 频谱图每个时间点的最大值索引
    window_size: 滑动窗口大小，用于分析连续性
    max_variance: 残差方差的最大阈值，小于此阈值被认为是连续的
    
    返回:
    tuple: (信号段列表, 连续性度量)
        - 信号段列表: 每个元素是检测到的连续信号的(起始索引,结束索引)
        - 连续性度量: 每个时间点对应的连续性度量（负方差值），
          其中continuity_metric[i]表示max_freq_indices[i:i+window_size]窗口的连续性
    """
    if len(max_freq_indices) < window_size:
        return [], np.zeros(len(max_freq_indices))
    
    # 计算连续性指标 - 长度为len(max_freq_indices) - window_size + 1
    continuity_metric = np.zeros(len(max_freq_indices) - window_size + 1)
    signal_segments = []
    
    # 在滑动窗口中计算连续性指标
    for i in range(len(max_freq_indices) - window_size + 1):
        window = max_freq_indices[i:i+window_size]
        x = np.arange(window_size).reshape(-1, 1)
        
        # 线性回归
        model = LinearRegression()
        model.fit(x, window)
        
        # 预测值和残差
        y_pred = model.predict(x)
        residuals = window - y_pred
        
        # 计算方差
        variance = np.var(residuals)
        
        # 使用负方差作为连续性指标（值越高表示越连续）
        continuity_metric[i] = -variance
    
    # 绘制连续性指标
    plt.figure(figsize=(12, 4))
    plt.plot(continuity_metric, label='连续性指标')
    plt.axhline(y=-max_variance, color='r', linestyle='--', label=f'阈值 (-{max_variance})')
    plt.grid(True)
    plt.xlabel('时间点')
    plt.ylabel('连续性指标 (负方差)')
    plt.title('信号连续性分析')
    plt.legend()
    plt.savefig('signal_continuity_detection.png')
    # plt.show()
    plt.close()
    # 检测连续区域（超过阈值的区域）
    is_signal = continuity_metric > -max_variance
    
    # 查找连续的信号段
    in_segment = False
    start_idx = 0
    
    for i in range(len(is_signal)):
        if is_signal[i] and not in_segment:
            in_segment = True
            start_idx = i
        elif not is_signal[i] and in_segment:
            in_segment = False
            if i - start_idx >= 1:  # 只保留足够长的段
                signal_segments.append((start_idx, i-1+window_size-1))  # 调整结束索引以对应原始数组
    
    # 检查最后一个段
    if in_segment:
        signal_segments.append((start_idx, len(max_freq_indices)-1))
    print(signal_segments)
    return signal_segments, continuity_metric


def correct_frequency_drift(wave_complex: np.ndarray, fs: float, sym_bin: float, sym_t: float, params=None):
    """
    对FT8信号进行频率漂移校正
    
    参数:
    wave_complex: 复数形式的信号，注意，只会取正频率部分用作频谱图
    fs: 采样率
    sym_bin: 符号频率间隔
    sym_t: 符号时间长度
    params: 可选参数字典，包含以下字段：
        - nsync_sym: 同步符号数量 (默认: 7)
        - ndata_sym: 数据符号数量 (默认: 58)
        - zscore_threshold: Z分数阈值 (默认: 5)
        - max_iteration_num: 最大迭代次数 (默认: 400)
        - debug_plots: 是否生成调试图 (默认: False)
        - window_size: 连续性分析的窗口大小 (默认: 8)
        - max_variance_factor: 残差方差阈值的因子 (默认: 0.0001)
          实际max_variance = max_variance_factor * freq_bins²
        - fit_middle_percent: 拟合时使用中间部分的百分比，头尾各去掉(100-fit_middle_percent)/2%
    
    返回:
    tuple: (校正后的信号, 估计的频率漂移率)
        - 校正后的信号: 采样率与输入信号相同(fs)的复数信号
        - 估计的频率漂移率: 每秒的频率漂移率(Hz/s)
    """
    # 设置默认参数
    default_params = {
        'nsync_sym': 7,
        'ndata_sym': 58,
        'zscore_threshold': 5,
        'max_iteration_num': 400,
        'debug_plots': True,
        'window_size': 8,
        'max_variance_factor': 0.0001,  # 方差因子，实际方差阈值将乘以频谱点数的平方
        'fit_middle_percent': 100,  # 拟合时使用中间部分的百分比，头尾各去掉(100-fit_middle_percent)/2%
        'bins_per_tone': 2,  # 每个音调的频率bin数量
        'steps_per_symbol': 2,  # 每个符号的时间步数
    }
    
    if params is None:
        params = default_params
    else:
        for key, value in default_params.items():
            if key not in params:
                params[key] = value

    # 设置matplotlib使用英文字体以避免中文字体问题
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 基本变量
    bins_per_tone = params['bins_per_tone']
    steps_per_symbol = params['steps_per_symbol']
    debug_plots = params['debug_plots']
    
    # 信号长度
    nsamples = len(wave_complex)
    tlength = nsamples / fs

    # 计算频谱图
    sx_filtered_db, f, t = calculate_spectrogram(
        wave_complex, fs, bins_per_tone, steps_per_symbol
    )
    
    # 只保留正频率部分
    positive_freq_mask = f >= 0
    sx_filtered_db = sx_filtered_db[positive_freq_mask]
    f = f[positive_freq_mask]
    
    # 创建FT8Waterfall对象
    waterfall = create_waterfall_from_spectrogram(
        sx_filtered_db, steps_per_symbol, bins_per_tone
    )
    
    # 基本变量
    time_osr = waterfall.time_osr
    freq_osr = waterfall.freq_osr
    
    # 从waterfall中获取频谱图数据
    sx_filtered_db = waterfall.mag
    
    # 频谱点数
    freq_bins = sx_filtered_db.shape[0]
    
    # 根据频谱点数的平方计算实际的方差阈值
    # 残差与频谱索引的范围平方成正比，因为索引可能在0到freq_bins-1之间变动
    max_variance = params['max_variance_factor'] * (freq_bins ** 2)
    
    if debug_plots:
        print(f"Spectrum bins: {freq_bins}, Variance threshold: {max_variance}")
    
    # 计算最大幅度频率-时间序列的索引
    max_freq_indices = np.zeros(sx_filtered_db.shape[1], dtype=int)
    for i in range(sx_filtered_db.shape[1]):
        max_freq_indices[i] = np.argmax(sx_filtered_db[:, i])
    
    # 先检测信号连续性
    signal_segments, continuity_metric = detect_signal_continuity(
        max_freq_indices, 
        window_size=params['window_size'],
        max_variance=max_variance  # 使用根据频谱点数计算的方差阈值
    )
    
    # 如果没有检测到信号段，返回原始信号
    if not signal_segments:
        print("No continuous signal segments detected, returning original signal")
        return wave_complex, 0.0
    
    # 选择最长的信号段
    longest_segment = max(signal_segments, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_segment
    
    # 只为检测到的信号段计算实际频率
    freq_step = sym_bin / freq_osr
    max_freqs = np.zeros(len(max_freq_indices))
    print("freq_step:", freq_step)

    # 计算整个序列的频率，但只用信号段的频率进行拟合
    for i in range(len(max_freq_indices)):
        max_freqs[i] = max_freq_indices[i] * freq_step
    
    # 计算时间轴（秒）
    time_step = sym_t / time_osr
    time_axis = np.arange(len(max_freqs)) * time_step
    
    # 绘制最大频率序列和连续性指标
    if debug_plots:
        plt.figure(figsize=(12, 8))
        
        # 主图：频率-时间曲线
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, max_freqs, label='Max Frequency', alpha=0.7)
        
        # 标记检测到的信号段
        for seg_start, seg_end in signal_segments:
            plt.axvspan(time_axis[seg_start], time_axis[seg_end], 
                       alpha=0.2, color='green', label='_' if seg_start > 0 else 'Detected Signal')
            # 高亮显示检测到的信号段的频率
            plt.plot(time_axis[seg_start:seg_end+1], max_freqs[seg_start:seg_end+1], 
                    'r-', linewidth=2, label='_')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Maximum Frequency Sequence and Detected Signal Segments')
        plt.grid(True)
        plt.legend()
        
        # 副图：连续性指标
        plt.subplot(2, 1, 2)
        # 创建连续性指标的时间轴（比原时间轴短window_size-1）
        continuity_time_axis = time_axis[:len(continuity_metric)]
        plt.plot(continuity_time_axis, continuity_metric, color='orange', label='Continuity Metric')
        
        # 标记阈值
        plt.axhline(y=-max_variance, color='r', linestyle='--', 
                   label=f'Variance Threshold ({max_variance})')
        
        # 标记检测到的信号段
        for seg_start, seg_end in signal_segments:
            # 调整结束索引，因为continuity_metric比原序列短
            adj_end = min(seg_end, len(continuity_metric)-1)
            if seg_start < len(continuity_metric):
                plt.axvspan(continuity_time_axis[seg_start], 
                           continuity_time_axis[min(adj_end, len(continuity_time_axis)-1)], 
                           alpha=0.2, color='green', label='_')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Continuity Metric (-variance)')
        plt.title('Signal Continuity Metric')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('signal_continuity_detection.png')
        # plt.show()
        plt.close()
    
    # 提取最长信号段的时间和频率数据用于估计频率漂移
    seg_time = time_axis[start_idx:end_idx+1].reshape(-1, 1)
    seg_freq = max_freqs[start_idx:end_idx+1]
    
    # 只使用中间部分的点进行拟合
    fit_middle_percent = params['fit_middle_percent']
    if fit_middle_percent < 100:
        trim_percent = (100 - fit_middle_percent) / 2 / 100  # 头尾各去掉的比例
        seg_len = len(seg_time)
        trim_points = int(seg_len * trim_percent)
        
        if trim_points > 0 and 2 * trim_points < seg_len:  # 确保有足够的点
            # 截取中间部分
            fit_start = trim_points
            fit_end = seg_len - trim_points
            seg_time_fit = seg_time[fit_start:fit_end]
            seg_freq_fit = seg_freq[fit_start:fit_end]
        else:
            # 点数不足或参数错误，使用全部点
            seg_time_fit = seg_time
            seg_freq_fit = seg_freq
    else:
        seg_time_fit = seg_time
        seg_freq_fit = seg_freq
    
    # 创建多项式特征和拟合
    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(seg_time_fit)
    
    # 多项式拟合
    model = LinearRegression()
    model.fit(X_poly, seg_freq_fit)
    
    # 获取系数
    intercept = model.intercept_  # 截距是标量
    coefs = model.coef_  # 多项式系数
    
    # 确保系数是一维数组
    if coefs.ndim > 1:
        coefs = coefs[0]
    
    # 获取多项式各项系数
    if len(coefs) > 2:  # 二次多项式
        raise ValueError("不支持二次多项式拟合，二次拟合结果有问题")
        f_shift_acc = coefs[2]  # 二次项系数 (Hz/s²)
        f_shift_rate = coefs[1]  # 一次项系数 (Hz/s)
    else:  # 线性多项式
        f_shift_acc = 0
        f_shift_rate = coefs[1] if len(coefs) > 1 else 0
    
    # 绘制频率漂移拟合结果
    if debug_plots:
        plt.figure(figsize=(10, 6))
        
        # 绘制原始数据点
        plt.scatter(seg_time, seg_freq, color='blue', label='Signal Frequency Points', alpha=0.3)
        
        # 标记用于拟合的点
        if fit_middle_percent < 100:
            plt.scatter(seg_time_fit, seg_freq_fit, color='green', label='Points Used for Fitting')
        
        # 生成平滑曲线用的x值
        x_smooth = np.linspace(seg_time.min(), seg_time.max(), 100).reshape(-1, 1)
        x_smooth_poly = poly.transform(x_smooth)
        
        # 计算拟合曲线的y值
        y_smooth = model.predict(x_smooth_poly)
        
        # 绘制拟合曲线
        plt.plot(x_smooth, y_smooth, color='red', label='Polynomial Fit')
        
        # 添加标签和标题
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Frequency Drift Fitting')
        plt.legend()
        plt.grid(True)
        
        # 显示拟合参数
        equation = f'Fit: f(t) = '
        if f_shift_acc != 0:
            equation += f'{f_shift_acc:.4f}t^2 + '
        equation += f'{f_shift_rate:.4f}t + {intercept:.2f}'
        
        plt.figtext(0.5, 0.01, equation, ha='center', fontsize=12)
        plt.figtext(0.5, 0.04, f'Drift Rate: {f_shift_rate:.4f} Hz/s, Accel: {f_shift_acc:.6f} Hz/s^2', ha='center', fontsize=12)
        
        # 保存图像
        plt.savefig('frequency_drift_fitting_new.png')
        plt.close()
    
    # 使用拟合模型直接预测频率偏移
    # 为每个样本点创建时间序列（秒）
    sample_time = np.arange(nsamples) / fs
    sample_time = sample_time.reshape(-1, 1)
    
    # 将时间序列转化为多项式特征
    sample_time_poly = poly.transform(sample_time)
    
    # 使用模型预测每个时间点的频率
    predicted_frequencies = model.predict(sample_time_poly)
    
    array_range = np.arange(nsamples)

    f_shift_est_k_hz_psample = f_shift_rate
    f_shift_est_k_hz_psample_acc = f_shift_acc
    print(f"f_shift_est_k_hz_psample: {f_shift_est_k_hz_psample}")
    print(f"f_shift_est_k_hz_psample_acc: {f_shift_est_k_hz_psample_acc}")

    compensation_carrier = np.exp(-2j*np.pi*(f_shift_est_k_hz_psample*array_range**2/2/fs+f_shift_est_k_hz_psample_acc*array_range**3/6/fs**2)/(fs))
    
    

    
    # 应用补偿
    wave_compensated = wave_complex * compensation_carrier
    
    if debug_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(sample_time, predicted_frequencies, label='Predicted Frequency Drift')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Predicted Frequency Drift Across Entire Signal')
        plt.grid(True)
        plt.legend()
        plt.savefig('predicted_frequency_drift.png')
        plt.close()

        
        print(f"Frequency drift rate: {f_shift_rate:.4f} Hz/s")
        print(f"Frequency drift acceleration: {f_shift_acc:.6f} Hz/s^2")
    
    # 为了保持与原来接口兼容，仅返回线性项系数
    return wave_compensated, f_shift_rate/fs
