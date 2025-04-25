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
                signal_segments.append((start_idx, i))  # 调整结束索引以对应原始数组
    
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
        - window_size_factor: 窗口大小因子，window_size = window_size_factor * steps_per_symbol (默认: 4)
        - max_variance_factor: 残差方差阈值的因子 (默认: 0.0001)
          实际max_variance = max_variance_factor * freq_bins²
        - fit_middle_percent: 拟合时使用中间部分的百分比，头尾各去掉(100-fit_middle_percent)/2%
        - poly_degree: 频率漂移拟合的多项式阶数 (默认: 1)
        - precise_sync: 是否进行精确时间同步 
    
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
        'window_size_factor': 4,  # 窗口大小因子，用于计算window_size
        'max_variance_factor': 0.0001,  # 方差因子，实际方差阈值将乘以频谱点数的平方
        'fit_middle_percent': 100,  # 拟合时使用中间部分的百分比，头尾各去掉(100-fit_middle_percent)/2%
        'bins_per_tone': 2,  # 每个音调的频率bin数量
        'steps_per_symbol': 2,  # 每个符号的时间步数
        'poly_degree': 2,      # 频率漂移拟合的多项式阶数
        'precise_sync': True,  # 是否进行精确时间同步
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
    nsync_sym = params['nsync_sym']
    ndata_sym = params['ndata_sym']
    zscore_threshold = params['zscore_threshold']
    max_iteration_num = params['max_iteration_num']
    
    # 计算window_size
    window_size = params['window_size_factor'] * steps_per_symbol

    
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
    
    # 先检测信号连续性 - 粗时间同步
    signal_segments, continuity_metric = detect_signal_continuity(
        max_freq_indices, 
        window_size=window_size,
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
        plt.close()
    
    # 提取最长信号段的时间和频率数据用于估计频率漂移
    seg_time = time_axis[start_idx:end_idx].reshape(-1, 1)
    seg_freq = max_freqs[start_idx:end_idx]
    
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
    
    # 获取线性多项式系数
    f_shift_rate = coefs[1] if len(coefs) > 1 else 0  # 线性项系数 (Hz/s)
    
    # 计算初步频率漂移补偿
    array_range = np.arange(nsamples)
    compensation_carrier_linear = np.exp(-2j*np.pi*(f_shift_rate*array_range**2/2/fs)/(fs))
    
    # 初步频率补偿
    wave_compensated_linear = wave_complex * compensation_carrier_linear
    
    # 如果不需要精确同步，直接返回线性补偿结果
    if not params['precise_sync']:
        return wave_compensated_linear, f_shift_rate/fs
    
    # ---- 精确时间同步 ----
    
    # 重新计算频谱图，使用初步矫正后的信号
    sx_filtered_db_2nd, f_2nd, t_2nd = calculate_spectrogram(
        wave_compensated_linear, fs, bins_per_tone, steps_per_symbol
    )
    
    # 只保留正频率部分
    positive_freq_mask_2nd = f_2nd >= 0
    sx_filtered_db_2nd = sx_filtered_db_2nd[positive_freq_mask_2nd]
    f_2nd = f_2nd[positive_freq_mask_2nd]
    
    # 创建第二个FT8Waterfall对象
    waterfall_2nd = create_waterfall_from_spectrogram(
        sx_filtered_db_2nd, steps_per_symbol, bins_per_tone
    )
    
    # 从第二个waterfall中获取频谱图数据
    sx_filtered_db_2nd = waterfall_2nd.mag
    
    # 计算最大幅度频率-时间序列的索引
    max_freq_indices_2nd = np.zeros(sx_filtered_db_2nd.shape[1], dtype=int)
    for i in range(sx_filtered_db_2nd.shape[1]):
        max_freq_indices_2nd[i] = np.argmax(sx_filtered_db_2nd[:, i])
    
    # 计算实际频率
    max_freqs_2nd = max_freq_indices_2nd * freq_step

    # 构造同步序列
    sync_seq = (np.array([3, 1, 4, 0, 6, 5, 2]) + 1)
    sync_seq = sync_seq - np.mean(sync_seq)

    # 每个符号的GFSK脉冲整形
    samples_per_sym = time_osr * 2
    t_pulse = np.linspace(-1, 1, samples_per_sym+1)
    gfsk_shape = gfsk_pulse(bt=2.0, t=t_pulse)

    # 扩展同步序列长度以适应整形脉冲
    sync_correlation_seq = np.zeros((nsync_sym-1) * time_osr + samples_per_sym + 1)

    # 对每个同步符号进行脉冲整形
    for sym_idx in range(nsync_sym):
        sync_correlation_seq[sym_idx * time_osr:(sym_idx * time_osr) + samples_per_sym + 1] += gfsk_shape * sync_seq[sym_idx]

    # 创建三个同步序列
    three_sync_correlation_seq = np.zeros((3*nsync_sym + ndata_sym - 1) * time_osr + 1 + samples_per_sym)


    for i in range(3):
        start_idx = i*(nsync_sym+ndata_sym//2)*time_osr
        end_idx = start_idx + len(sync_correlation_seq)
        three_sync_correlation_seq[start_idx:end_idx] = sync_correlation_seq
        # 绘制three_sync_correlation_seq序列

    # 修改：只在连续信号段内进行相关计算
    correlation_peak_index = 0
    max_correlation_value = 0
    # 选择最长的信号段
    longest_segment = max(signal_segments, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_segment

    # TODO: 修正end_idx
    end_idx = end_idx + window_size - 2
    
    # 只在检测到的连续信号段内进行相关计算
    # 创建修改后的频率序列，只保留连续段，其余部分置零
    max_freqs_masked = np.zeros_like(max_freqs_2nd)

    max_freqs_masked[start_idx:end_idx] = max_freqs_2nd[start_idx:end_idx]

    max_freqs_masked[start_idx:end_idx] = max_freqs_masked[start_idx:end_idx] - np.mean(max_freqs_masked[start_idx:end_idx])
    # 相关计算 - 只在连续信号段上进行
    sync_correlation = np.correlate(max_freqs_masked, three_sync_correlation_seq, mode='full')

    # 绘制同步相关序列
    if debug_plots:
        print(longest_segment)
        plt.figure(figsize=(12, 6))
        plt.plot(max_freqs_masked, label='Max Freqs Masked Sequence')
        plt.title('Max Freqs Masked Sequence')
        plt.xlabel('Sample Points')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.savefig('max_freqs_masked.png')
        plt.close()

    # 绘制同步相关序列
    if debug_plots:
        print(longest_segment)
        plt.figure(figsize=(12, 6))
        plt.plot(sync_correlation, label='Sync Correlation Sequence')
        plt.title('Sync Correlation Sequence')
        plt.xlabel('Sample Points')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.savefig('sync_correlation.png')
        plt.close()

    # 找到相关峰值
    correlation_peak_index = np.argmax(sync_correlation)
    correlation_peak_time_block_index = correlation_peak_index - (len(three_sync_correlation_seq) - 1) + samples_per_sym//2
    


    
    # 可视化精确同步结果
    if debug_plots:
        plt.figure(figsize=(12, 6))
        plt.plot(sync_correlation, label='Sync Correlation')
        plt.axvline(x=correlation_peak_index, color='r', linestyle='--', label='Peak')
        plt.title('Precise Time Synchronization (using signal segment only)')
        plt.xlabel('Correlation Lag')
        plt.ylabel('Correlation Value')
        plt.grid(True)
        plt.legend()
        plt.savefig('precise_sync_correlation.png')
        plt.close()
        
        # 绘制找到的精确同步点对应的频率轨迹
        plt.figure(figsize=(12, 6))
        t_axis_2nd = np.arange(len(max_freqs_2nd)) * time_step
        plt.plot(t_axis_2nd, max_freqs_2nd, label='Frequency Trajectory', alpha=0.5)
        plt.plot(t_axis_2nd, max_freqs_masked, label='Masked Trajectory (Signal Segment)', alpha=0.8)
        plt.axvline(x=correlation_peak_time_block_index * time_step, color='r', linestyle='--', 
                   label=f'Precise Sync Point (t={correlation_peak_time_block_index * time_step:.3f}s)')
        # 标记信号段区域
        plt.axvspan(start_idx * time_step, end_idx * time_step, 
                  alpha=0.2, color='green', label='Signal Segment')
        plt.title('Precise Synchronization on Frequency Trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True)
        plt.legend()
        plt.savefig('precise_sync_point.png')
        plt.close()
    
    # ---- 高次频偏估计 ----
    
    # 根据精确同步点提取同步序列区域进行高次频偏估计
    regression_x = np.array([])
    regression_y = np.array([])

    # 从三个同步序列位置提取数据点
    for i in range(3):
        start_idx = i*(nsync_sym+ndata_sym//2)*time_osr + correlation_peak_time_block_index
        end_idx = start_idx + (nsync_sym-1) * time_osr
        
        # 确保索引在有效范围内
        if start_idx < len(max_freqs_masked):
            x_step = sym_t/time_osr
            x_values = np.arange(start_idx, min(end_idx, len(max_freqs_masked))) * x_step
            y_values = max_freqs_masked[start_idx:min(end_idx, len(max_freqs_masked))]
            
            # 添加到回归数据中
            regression_x = np.append(regression_x, x_values)
            regression_y = np.append(regression_y, y_values)
    
    # 如果数据点不足，使用整个信号段
    if len(regression_x) < 10:
        print("Not enough sync points found, using the entire signal segment!!!!!!!!!!!!!、")
        return wave_compensated_linear, f_shift_rate/fs
    
    # 使用高次多项式进行频偏建模
    poly_degree_final = params['poly_degree']
    
    # 确保有足够的数据点进行高次拟合
    if len(regression_x) > poly_degree_final + 1:
        # 准备数据
        regression_x = regression_x.reshape(-1, 1)
        
        # 创建多项式特征
        poly_final = PolynomialFeatures(degree=poly_degree_final)
        X_poly_final = poly_final.fit_transform(regression_x)
        
        # 拟合模型
        model_final = LinearRegression()
        model_final.fit(X_poly_final, regression_y)
        
        # 获取系数
        intercept_final = model_final.intercept_
        coefs_final = model_final.coef_
        
        # 确保系数是一维数组
        if coefs_final.ndim > 1:
            coefs_final = coefs_final[0]
        
        # 提取各阶系数
        f_shift_rate_final = coefs_final[1] if len(coefs_final) > 1 else 0.0  # 一次项系数 (Hz/s)
        f_shift_acc_final = coefs_final[2] if len(coefs_final) > 2 else 0.0   # 二次项系数 (Hz/s²)
        
        # 可视化高次拟合结果
        if debug_plots:
            plt.figure(figsize=(10, 6))
            
            # 绘制用于拟合的数据点
            plt.scatter(regression_x, regression_y, color='blue', label='Sync Points', alpha=0.5)
            
            # 生成平滑曲线
            x_smooth = np.linspace(regression_x.min(), regression_x.max(), 100).reshape(-1, 1)
            X_smooth_poly = poly_final.transform(x_smooth)
            y_smooth = model_final.predict(X_smooth_poly)
            
            # 绘制拟合曲线
            plt.plot(x_smooth, y_smooth, color='red', label=f'Degree-{poly_degree_final} Polynomial Fit')
            
            # 添加标签和标题
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('High-Order Frequency Drift Fitting')
            plt.legend()
            plt.grid(True)
            
            # 显示拟合方程
            equation = f'f(t) = '
            for i in range(poly_degree_final, -1, -1):
                if i > 0:
                    coef = coefs_final[i] if i < len(coefs_final) else 0
                    if coef != 0:
                        equation += f'{coef:.4e}t^{i} + '
                else:
                    equation += f'{intercept_final:.4f}'
            
            plt.figtext(0.5, 0.01, equation, ha='center', fontsize=10)
            
            if poly_degree_final >= 2:
                info = f'Linear rate: {f_shift_rate_final:.4f} Hz/s, Acceleration: {f_shift_acc_final:.4e} Hz/s²'
                plt.figtext(0.5, 0.04, info, ha='center', fontsize=10)
            else:
                info = f'Linear rate: {f_shift_rate_final:.4f} Hz/s'
                plt.figtext(0.5, 0.04, info, ha='center', fontsize=10)
            
            plt.savefig('high_order_drift_fitting.png')
            plt.close()
        
        # 计算频率偏移补偿
        compensation_carrier_final = np.ones(nsamples, dtype=complex)
        
        if poly_degree_final == 1:
            # 对于线性频偏，频移量与时间的平方成正比
            # 线性频率漂移的相位累积是时间的三次方
            compensation_carrier_final = np.exp(-2j*np.pi*f_shift_rate_final*array_range**2/(2*fs**2))
        elif poly_degree_final == 2:
            # 二阶多项式拟合情况下：
            # 频移量 = f_shift_rate_final*t + f_shift_acc_final*t^2
            # 相位累积 = f_shift_rate_final*t^2/2 + f_shift_acc_final*t^3/3
            # 对于离散序列，t = array_range/fs
            t = array_range/fs
            phase_accumulation = f_shift_rate_final*t**2/2 + f_shift_acc_final*t**3/3
            compensation_carrier_final = np.exp(-2j*np.pi*phase_accumulation)
            
            # 绘制补偿载波的频谱图
            if debug_plots:
                plt.figure(figsize=(10, 6))
                compensation_carrier_final_db, f, t = calculate_spectrogram(
                    compensation_carrier_final, fs, bins_per_tone, steps_per_symbol
                )
                
                plt.imshow(compensation_carrier_final_db, aspect='auto', origin='lower', 
                   extent=[t[0], t[-1], f[0], f[-1]])
                plt.colorbar(label='Magnitude (dB)')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Corrected Signal Spectrogram')
                plt.grid(True)
                plt.savefig('compensation_carrier_spectrogram.png')
                plt.close()
        else:
            print("poly_degree_final is not 1 or 2, using linear drift correction")
            return wave_compensated_linear, f_shift_rate/fs
        
        # 应用最终补偿
        wave_compensated_final = wave_compensated_linear * compensation_carrier_final
        
        # 返回结果和参数
        drift_params = {
            'drift_rate': f_shift_rate_final,
            'drift_acc': f_shift_acc_final,
            'sync_time': correlation_peak_time_block_index * time_step
        }
        
        # 如果有更高阶系数，也添加到结果中
        for i in range(3, poly_degree_final + 1):
            if i < len(coefs_final):
                drift_params[f'coef_order_{i}'] = coefs_final[i]
        
        print(f"Final drift parameters: {drift_params}")
        return wave_compensated_final, f_shift_rate_final/fs
    
    # 如果数据点不足进行高次拟合，退回到线性拟合结果
    print("Not enough data for high-order fitting, using linear drift correction")
    return wave_compensated_linear, f_shift_rate/fs
