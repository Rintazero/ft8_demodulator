import numpy as np
import scipy
import matplotlib.pyplot as plt
import logging
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from ..ft8_demodulator.ftx_types import FT8Waterfall
from sklearn.preprocessing import PolynomialFeatures
from ..ft8_demodulator.spectrogram_analyse import calculate_spectrogram
from ..ft8_demodulator.ft8_decode import create_waterfall_from_spectrogram

# 设置全局字体大小变量
FONT_SIZE = 16  # 调大的字体大小

# 配置模块级别的日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    print("Adding console handler")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

# 设置matplotlib字体设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用DejaVu Sans以获得更好的兼容性
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
plt.rcParams['font.size'] = FONT_SIZE  # 全局字体大小
plt.rcParams['axes.labelsize'] = FONT_SIZE  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = FONT_SIZE + 2  # 标题字体大小
plt.rcParams['xtick.labelsize'] = FONT_SIZE - 2  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = FONT_SIZE - 2  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = FONT_SIZE - 2  # 图例字体大小
plt.rcParams['figure.titlesize'] = FONT_SIZE + 4  # 图形标题字体大小

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
    # 创建一个包含两个子图的图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制连续性指标图像
    ax1.plot(continuity_metric)
    ax1.axhline(y=-max_variance, color='r', linestyle='--', label='阈值')
    ax1.set_xlabel('时间点')
    ax1.set_ylabel('连续性指标')
    ax1.set_title('信号连续性指标')
    ax1.legend()
    ax1.grid(True)
    
    # 标记检测到的信号段
    for start, end in signal_segments:
        ax1.axvspan(start, end, color='green', alpha=0.2)
    
    # 绘制最大频率轨迹
    ax2.plot(max_freq_indices, label='频率轨迹')
    ax2.set_xlabel('时间点')
    ax2.set_ylabel('频率索引')
    ax2.set_title('信号频率轨迹')
    ax2.grid(True)
    
    # 标记检测到的信号段
    for start, end in signal_segments:
        ax2.axvspan(start, end, color='green', alpha=0.2)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('signal_analysis.png')
    plt.show()
    plt.close()
    logger.debug("Detected signal segments: %s", signal_segments)
    return signal_segments, continuity_metric

def remove_outliers_iterative(freq_indices, spectrogram, zscore_threshold=5.0, max_iter=None, debug_plots=False, font_size=16):
    """
    使用迭代线性回归和Z-score方法移除频率轨迹中的离群点（模仿初代算法）
    
    参数:
    freq_indices: 频率索引数据（对应频谱图的行索引）
    spectrogram: 频谱图数据，形状为(频率bins, 时间steps)
    zscore_threshold: Z-score阈值，超过此值的点被视为离群点
    max_iter: 最大迭代次数，如果为None则设置为len(freq_indices)的100%
    debug_plots: 是否生成调试图
    font_size: 字体大小
    
    返回:
    cleaned_freq_indices: 清除离群点后的频率索引
    """
    if max_iter is None:
        max_iter = max(int(len(freq_indices) * 1), 1)  # 默认最大迭代次数为数据长度的10%
    
    # 复制原始数据
    cleaned_freq_indices = freq_indices.copy().astype(int)
    cleaned_spectrogram = spectrogram.copy()
    
    # 记录被修正的点
    corrected_points = []
    
    # 迭代处理离群点
    iteration_num = 0
    
    while iteration_num < max_iter:
        # 使用频率索引进行线性回归（模仿初代算法）
        x = np.arange(len(cleaned_freq_indices)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, cleaned_freq_indices)
        freq_indices_fitted = model.predict(x)
        
        # 计算残差
        residuals = cleaned_freq_indices - freq_indices_fitted
        
        # 使用scipy.stats.zscore计算Z分数（模仿初代算法）
        z_scores = np.abs(stats.zscore(residuals))
        
        # 找到所有超过阈值的离群点
        outlier_indices = np.where(z_scores > zscore_threshold)[0]
        
        # 如果没有离群点，结束迭代
        if len(outlier_indices) == 0:
            break
        
        # 选择Z分数最大的离群点进行处理（模仿初代算法）
        max_zscore_idx_in_outliers = np.argmax(z_scores[outlier_indices])
        max_zscore_idx = outlier_indices[max_zscore_idx_in_outliers]
        
        # 记录离群点的原始值
        original_freq_idx = cleaned_freq_indices[max_zscore_idx]
        
        # 将该离群点在频谱图中的值设为最小值（模仿初代算法）
        cleaned_spectrogram[original_freq_idx, max_zscore_idx] = np.min(cleaned_spectrogram)
        
        # 重新计算该时间点的最大频率索引（模仿初代算法）
        new_freq_idx = np.argmax(cleaned_spectrogram[:, max_zscore_idx])
        
        # 记录该离群点的修正信息
        corrected_points.append((max_zscore_idx, original_freq_idx, new_freq_idx))
        
        # 更新频率索引
        cleaned_freq_indices[max_zscore_idx] = new_freq_idx
        
        iteration_num += 1
    
    if iteration_num == max_iter:
        logger.warning("Outlier removal iteration reached max_iter, max z_score: %f", max(z_scores))
    
    logger.debug("max z_score: %f", max(z_scores))
    # 如果需要生成调试图
    if debug_plots and corrected_points:
        plt.figure(figsize=(12, 6))
        
        # 绘制原始数据
        time_points = np.arange(len(freq_indices))
        plt.plot(time_points, freq_indices, 'b-', alpha=0.5, label='原始频率索引轨迹')
        
        # 绘制清理后的数据
        plt.plot(time_points, cleaned_freq_indices, 'g-', label='清理后的频率索引轨迹')
        
        # 标记被修正的离群点
        corrected_x = [time_points[idx] for idx, _, _ in corrected_points]
        corrected_orig_y = [orig_y for _, orig_y, _ in corrected_points]
        corrected_new_y = [new_y for _, _, new_y in corrected_points]
        
        plt.scatter(corrected_x, corrected_orig_y, color='red', s=50, marker='x', label='检测到的离群点')
        plt.scatter(corrected_x, corrected_new_y, color='orange', s=50, marker='o', label='替换值（重新计算的最大值）')
        
        # 用箭头连接原始点和修正点
        for i in range(len(corrected_points)):
            plt.annotate('', 
                        xy=(corrected_x[i], corrected_new_y[i]),
                        xytext=(corrected_x[i], corrected_orig_y[i]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        plt.xlabel('时间点索引', fontsize=font_size)
        plt.ylabel('频率索引', fontsize=font_size)
        plt.title(f'离群点修正 (阈值Z={zscore_threshold}, 修正了{len(corrected_points)}个点)', fontsize=font_size+2)
        plt.grid(True)
        plt.legend(fontsize=font_size-2)
        plt.savefig('outlier_correction.png')
        plt.close()
    
    return cleaned_freq_indices

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
        - debug_plots: 是否生成调试图 (默认: False)
        - window_size_factor: 窗口大小因子，window_size = window_size_factor * steps_per_symbol (默认: 4)
        - max_variance_factor: 残差方差阈值的因子 (默认: 0.0001)
          实际max_variance = max_variance_factor * freq_bins²
        - fit_middle_percent: 拟合时使用中间部分的百分比，头尾各去掉(100-fit_middle_percent)/2%
        - poly_degree: 频率漂移拟合的多项式阶数 (默认: 1)
        - precise_sync: 是否进行精确时间同步 
        - outlier_zscore_threshold: 离群点Z-score阈值 (默认: 5.0)
        - outlier_max_iter_factor: 离群点最大迭代次数与time_osr的比例因子 (默认: 2.0)
    
    返回:
    tuple: (校正后的信号, 估计的频率漂移率)
        - 校正后的信号: 采样率与输入信号相同(fs)的复数信号
        - 估计的频率漂移率: 每秒的频率漂移率(Hz/s)
    """
    # 设置默认参数
    default_params = {
        'nsync_sym': 7,
        'ndata_sym': 58,
        'debug_plots': True,
        'window_size_factor': 4,  # 窗口大小因子，用于计算window_size
        'max_variance_factor': 0.0001,  # 方差因子，实际方差阈值将乘以频谱点数的平方
        'fit_middle_percent': 100,  # 拟合时使用中间部分的百分比，头尾各去掉(100-fit_middle_percent)/2%
        'bins_per_tone': 2,  # 每个音调的频率bin数量
        'steps_per_symbol': 2,  # 每个符号的时间步数
        'poly_degree': 2,      # 频率漂移拟合的多项式阶数
        'precise_sync': True,  # 是否进行精确时间同步
        'font_size': FONT_SIZE,  # 添加字体大小参数
        'outlier_zscore_threshold': 5.0,  # 离群点Z-score阈值
        'outlier_max_iter_factor': 1000.0,   # 离群点最大迭代次数与time_osr的比例因子
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
    plt.rcParams['font.size'] = params['font_size']  # 使用参数中的字体大小
    plt.rcParams['axes.labelsize'] = params['font_size']
    plt.rcParams['axes.titlesize'] = params['font_size'] + 2
    plt.rcParams['xtick.labelsize'] = params['font_size'] - 2
    plt.rcParams['ytick.labelsize'] = params['font_size'] - 2
    plt.rcParams['legend.fontsize'] = params['font_size'] - 2
    plt.rcParams['figure.titlesize'] = params['font_size'] + 4

    # 基本变量
    bins_per_tone = params['bins_per_tone']
    steps_per_symbol = params['steps_per_symbol']
    debug_plots = params['debug_plots']
    nsync_sym = params['nsync_sym']
    ndata_sym = params['ndata_sym']
    
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
    # positive_freq_mask = f >= 0
    # sx_filtered_db = sx_filtered_db[positive_freq_mask]
    # f = f[positive_freq_mask]
    
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
    
    if debug_plots:
        # Plot spectrogram
        plt.figure(figsize=(10, 6))
        plt.imshow(sx_filtered_db, aspect='auto', origin='lower',
                extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (s)', fontsize=params['font_size'])
        plt.ylabel('Frequency (Hz)', fontsize=params['font_size']) 
        plt.title('Signal Spectrogram', fontsize=params['font_size']+2)
        plt.grid(True)
        plt.savefig('signal_spectrogram.png')
        # plt.show()
        plt.close()
    
    # 根据频谱点数的平方计算实际的方差阈值
    # 残差与频谱索引的范围平方成正比，因为索引可能在0到freq_bins-1之间变动
    max_variance = params['max_variance_factor'] * (freq_bins ** 2)
    
    if debug_plots:
        logger.debug("Spectrum bins: %d, Variance threshold: %f", freq_bins, max_variance)
    
    # 计算最大幅度频率-时间序列的索引
    max_freq_indices = np.zeros(sx_filtered_db.shape[1], dtype=int)
    for i in range(sx_filtered_db.shape[1]):
        max_freq_indices[i] = np.argmax(sx_filtered_db[:, i])
    
    # 计算时间轴（秒）
    time_step = sym_t / time_osr
    time_axis = np.arange(len(max_freq_indices)) * time_step
    
    # 计算频率轴（Hz）
    freq_step = sym_bin / freq_osr
    max_freqs = max_freq_indices * freq_step
    
    # 先检测信号连续性 - 粗时间同步
    signal_segments, continuity_metric = detect_signal_continuity(
        max_freq_indices, 
        window_size=window_size,
        max_variance=max_variance  # 使用根据频谱点数计算的方差阈值
    )
    
    # 如果没有检测到信号段，返回原始信号
    if not signal_segments:
        logger.warning("No continuous signal segments detected, returning original signal")
        return wave_complex, 0.0
    
    # 选择最长的信号段
    longest_segment = max(signal_segments, key=lambda x: x[1] - x[0])
    seg_start_idx, seg_end_idx = longest_segment
    
    # 第一次离群点修正 - 在信号段检测后，使用检测到的信号段进行切分
    outlier_max_iter = int(params['outlier_max_iter_factor'] * time_osr)
    
    # 对频谱图和索引进行切分
    if seg_start_idx < len(max_freq_indices) and seg_end_idx <= len(max_freq_indices):
        # 切分频率索引序列
        max_freq_indices_segment = max_freq_indices[seg_start_idx:seg_end_idx]
        # 切分频谱图
        sx_filtered_db_segment = sx_filtered_db[:, seg_start_idx:seg_end_idx]
        
        # 在切分的数据上进行离群点修正
        max_freq_indices_segment_cleaned = remove_outliers_iterative(
            max_freq_indices_segment,
            sx_filtered_db_segment,
            zscore_threshold=params['outlier_zscore_threshold'],
            max_iter=outlier_max_iter,
            debug_plots=debug_plots,
            font_size=params['font_size']
        )
        
        # 将清理后的结果替换回原序列
        max_freq_indices[seg_start_idx:seg_end_idx] = max_freq_indices_segment_cleaned
    else:
        # 如果索引超出范围，使用完整序列
        logger.warning("Signal segment indices out of range, using full sequence for first outlier removal")
        max_freq_indices = remove_outliers_iterative(
            max_freq_indices,
            sx_filtered_db,
            zscore_threshold=params['outlier_zscore_threshold'],
            max_iter=outlier_max_iter,
            debug_plots=debug_plots,
            font_size=params['font_size']
        )
    
    # 重新计算频率数据
    max_freqs = max_freq_indices * freq_step
    
    if debug_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, max_freqs, 'b-', label='Maximum Frequency (After First Outlier Removal)')
        plt.xlabel('Time (s)', fontsize=params['font_size'])
        plt.ylabel('Frequency (Hz)', fontsize=params['font_size'])
        plt.title('Frequency vs Time (After First Outlier Removal)', fontsize=params['font_size']+2)
        plt.grid(True)
        plt.legend(fontsize=params['font_size']-2)
        plt.savefig('frequency_vs_time_cleaned.png')
        # plt.show()
        plt.close()

    # 重新计算实际频率（用于后续处理）
    freq_step = sym_bin / freq_osr
    max_freqs = np.zeros(len(max_freq_indices))
    logger.debug("freq_step: %f", freq_step)

    # 计算整个序列的频率，但只用信号段的频率进行拟合
    for i in range(len(max_freq_indices)):
        max_freqs[i] = max_freq_indices[i] * freq_step
    
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
        
        plt.xlabel('Time (s)', fontsize=params['font_size'])
        plt.ylabel('Frequency (Hz)', fontsize=params['font_size'])
        plt.title('Maximum Frequency Sequence and Detected Signal Segments', fontsize=params['font_size']+2)
        plt.grid(True)
        plt.legend(fontsize=params['font_size']-2)
        
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
        
        plt.xlabel('Time (s)', fontsize=params['font_size'])
        plt.ylabel('Continuity Metric (-variance)', fontsize=params['font_size'])
        plt.title('Signal Continuity Metric', fontsize=params['font_size']+2)
        plt.grid(True)
        plt.legend(fontsize=params['font_size']-2)
        
        plt.tight_layout()
        plt.savefig('signal_continuity_detection.png')
        plt.show()
        plt.close()
    
    # 提取最长信号段的时间和频率数据用于估计频率漂移
    seg_time = time_axis[seg_start_idx:seg_end_idx].reshape(-1, 1)
    seg_freq = max_freqs[seg_start_idx:seg_end_idx]
    
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
    
    # 绘制拟合结果
    if debug_plots:
        plt.figure(figsize=(10, 6))
        
        # Plot data points used for fitting
        plt.scatter(seg_time_fit, seg_freq_fit, color='green', alpha=0.5, label='Points Used for Fitting')
        
        # Generate fitting curve
        x_smooth = np.linspace(seg_time.min(), seg_time.max(), 100).reshape(-1, 1)
        X_smooth_poly = poly.transform(x_smooth)
        y_smooth = model.predict(X_smooth_poly)
        
        # Plot fitting curve
        plt.plot(x_smooth, y_smooth, color='red', label='Linear Fit')
        
        plt.xlabel('Time (s)', fontsize=params['font_size'])
        plt.ylabel('Frequency (Hz)', fontsize=params['font_size'])
        plt.title('Frequency Drift Linear Fitting', fontsize=params['font_size']+2)
        plt.legend(fontsize=params['font_size']-2)
        plt.grid(True)
        
        # Display fitting equation
        equation = f'f(t) = {model.coef_[1]:.4f}t + {model.intercept_:.4f}'
        plt.figtext(0.5, 0.02, equation, ha='center', fontsize=params['font_size']-2)
        
        plt.savefig('frequency_drift_fitting.png')
        # plt.show()
        plt.close()
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
    
    if debug_plots:
        plt.figure(figsize=(10, 6))
        plt.imshow(sx_filtered_db_2nd, aspect='auto', origin='lower', extent=[t_2nd[0], t_2nd[-1], f_2nd[0], f_2nd[-1]])
        plt.colorbar(label='Intensity (dB)')
        plt.title('FT8 Signal Spectrogram', fontsize=params['font_size']+2)
        plt.xlabel('Time (s)', fontsize=params['font_size'])
        plt.ylabel('Frequency (Hz)', fontsize=params['font_size'])
        plt.savefig('corrected_signal_spectrogram_second.png')
        plt.close()
    

    # 只保留正频率部分
    # positive_freq_mask_2nd = f_2nd >= 0
    # sx_filtered_db_2nd = sx_filtered_db_2nd[positive_freq_mask_2nd]
    # f_2nd = f_2nd[positive_freq_mask_2nd]
    
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

    # 第二次离群点修正 - 根据检测到的信号段切分频谱图
    time_axis_2nd = np.arange(len(max_freqs_2nd)) * time_step
    
    # 对频谱图和索引进行切分（使用相同的信号段）
    # 切分频率索引序列
    max_freq_indices_2nd_segment = max_freq_indices_2nd[seg_start_idx:seg_end_idx]
    # 切分频谱图
    sx_filtered_db_2nd_segment = sx_filtered_db_2nd[:, seg_start_idx:seg_end_idx]
    
    # 在切分的数据上进行离群点修正
    max_freq_indices_2nd_segment_cleaned = remove_outliers_iterative(
        max_freq_indices_2nd_segment,
        sx_filtered_db_2nd_segment,
        zscore_threshold=params['outlier_zscore_threshold'],
        max_iter=outlier_max_iter,
        debug_plots=debug_plots,
        font_size=params['font_size']
    )
    
    # 将清理后的结果替换回原序列
    max_freq_indices_2nd[seg_start_idx:seg_end_idx] = max_freq_indices_2nd_segment_cleaned
    
    # 重新计算频率数据
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
    seg_start_idx, seg_end_idx = longest_segment

    # TODO: 修正end_idx
    end_idx = seg_end_idx + window_size - 3
    # 修正start_idx
    start_idx = seg_start_idx
    
    # 只在检测到的连续信号段内进行相关计算
    # 创建修改后的频率序列，只保留连续段，其余部分置零
    max_freqs_masked = np.zeros_like(max_freqs_2nd)

    max_freqs_masked[seg_start_idx:seg_end_idx] = max_freqs_2nd[seg_start_idx:seg_end_idx]

    max_freqs_masked[seg_start_idx:seg_end_idx] = max_freqs_masked[seg_start_idx:seg_end_idx] - np.mean(max_freqs_masked[seg_start_idx:seg_end_idx])
    # 相关计算 - 只在连续信号段上进行
    sync_correlation = np.correlate(max_freqs_masked, three_sync_correlation_seq, mode='full')

    # 绘制同步相关序列
    if debug_plots:
        logger.debug("Detected longest segment: %s", longest_segment)
        plt.figure(figsize=(12, 6))
        plt.plot(max_freqs_masked, label='Max Freqs Masked Sequence')
        plt.title('Max Freqs Masked Sequence', fontsize=params['font_size']+2)
        plt.xlabel('Sample Points', fontsize=params['font_size'])
        plt.ylabel('Amplitude', fontsize=params['font_size'])
        plt.grid(True)
        plt.legend(fontsize=params['font_size']-2)
        plt.savefig('max_freqs_masked.png')
        # plt.show()
        plt.close()

    
    # 绘制同步序列
    if debug_plots:
        logger.debug("Plotting sync correlation sequence")
        plt.figure(figsize=(12, 6))
        plt.plot(three_sync_correlation_seq, label='Sync Sequence')
        plt.title('Sync Sequence', fontsize=params['font_size']+2)
        plt.xlabel('Sample Points', fontsize=params['font_size'])
        plt.ylabel('Amplitude', fontsize=params['font_size'])
        plt.grid(True)
        plt.legend(fontsize=params['font_size']-2)
        plt.savefig('three_sync_correlation_seq.png')
        # plt.show()
        plt.close()

    # 找到相关峰值
    correlation_peak_index = np.argmax(sync_correlation)
    

    correlation_peak_time_block_index = correlation_peak_index - (len(three_sync_correlation_seq) - 1) + samples_per_sym//2
    


    
    # 可视化精确同步结果
    if debug_plots:
        plt.figure(figsize=(12, 6))
        plt.plot(sync_correlation, label='Sync Correlation')
        plt.axvline(x=correlation_peak_index, color='r', linestyle='--', label='Peak')
        plt.title('Precise Time Synchronization', fontsize=params['font_size']+2)
        plt.xlabel('Correlation Lag', fontsize=params['font_size'])
        plt.ylabel('Correlation Value', fontsize=params['font_size'])
        plt.grid(True)
        plt.legend(fontsize=params['font_size']-2)
        plt.savefig('precise_sync_correlation.png')
        # plt.show()
        plt.close()
        
        # 绘制找到的精确同步点对应的频率轨迹
        plt.figure(figsize=(12, 6))
        t_axis_2nd = np.arange(len(max_freqs_2nd)) * time_step
        plt.plot(t_axis_2nd, max_freqs_2nd, label='Frequency Trajectory', alpha=0.5)
        plt.plot(t_axis_2nd, max_freqs_masked, label='Masked Trajectory (Signal Segment)', alpha=0.8)
        plt.axvline(x=correlation_peak_time_block_index * time_step, color='r', linestyle='--', 
                   label=f'Precise Sync Point (t={correlation_peak_time_block_index * time_step:.3f}s)')
        # 标记信号段区域
        plt.axvspan(seg_start_idx * time_step, seg_end_idx * time_step, 
                  alpha=0.2, color='green', label='Signal Segment')
        plt.title('Precise Synchronization on Frequency Trajectory', fontsize=params['font_size']+2)
        plt.xlabel('Time (s)', fontsize=params['font_size'])
        plt.ylabel('Frequency (Hz)', fontsize=params['font_size'])
        plt.grid(True)
        plt.legend(fontsize=params['font_size']-2)
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
    
    # 绘制max_freqs_masked序列和三个同步序列提取的数据点
    if debug_plots:
        plt.figure(figsize=(14, 8))
        
        # 计算时间轴
        time_axis_masked = np.arange(len(max_freqs_masked)) * time_step
        
        # 绘制完整的max_freqs_masked序列
        plt.plot(time_axis_masked, max_freqs_masked, color='darkblue', alpha=0.8, linewidth=1.5, label='Max Freqs Masked Sequence')
        
        # 标注三个同步序列位置提取的数据点
        colors = ['red', 'green', 'orange']
        for i in range(3):
            start_idx = i*(nsync_sym+ndata_sym//2)*time_osr + correlation_peak_time_block_index
            end_idx = start_idx + (nsync_sym-1) * time_osr
            
            # 确保索引在有效范围内
            if start_idx < len(max_freqs_masked):
                # 计算实际的结束索引
                actual_end_idx = min(end_idx, len(max_freqs_masked))
                
                # 提取时间和频率数据
                time_indices = np.arange(start_idx, actual_end_idx)
                time_values = time_indices * time_step
                freq_values = max_freqs_masked[start_idx:actual_end_idx]
                
                # 绘制提取的数据点
                plt.scatter(time_values, freq_values, color=colors[i], s=20, alpha=0.8, 
                           label=f'Sync Sequence {i+1} Data Points')
                
                # 标注区域范围
                plt.axvspan(start_idx * time_step, (actual_end_idx-1) * time_step, 
                           alpha=0.15, color=colors[i])
                
                # 添加起始位置的垂直线
                plt.axvline(x=start_idx * time_step, color=colors[i], linestyle='--', 
                           alpha=0.6, linewidth=1)
        
        # 标注精确同步点
        plt.axvline(x=correlation_peak_time_block_index * time_step, color='black', 
                   linestyle='-', linewidth=2, alpha=0.8, label='Precise Sync Point')
        
        plt.xlabel('Time (s)', fontsize=params['font_size'])
        plt.ylabel('Frequency (Hz)', fontsize=params['font_size'])
        plt.title('Max Freqs Masked Sequence with Sync Data Points', fontsize=params['font_size']+2)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=params['font_size']-2, loc='best')
        
        plt.tight_layout()
        plt.savefig('sync_data_points_extraction.png', dpi=150, bbox_inches='tight')
        # plt.show()
        plt.close()
    
    # 如果数据点不足，使用整个信号段
    if len(regression_x) < 10:
        logger.warning("Not enough sync points found, using the entire signal segment")
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
            plt.xlabel('Time (s)', fontsize=params['font_size'])
            plt.ylabel('Frequency (Hz)', fontsize=params['font_size'])
            plt.title('High-Order Frequency Drift Fitting', fontsize=params['font_size']+2)
            plt.legend(fontsize=params['font_size']-2)
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
            
            plt.figtext(0.5, 0.01, equation, ha='center', fontsize=params['font_size']-2)
            
            if poly_degree_final >= 2:
                info = f'Linear rate: {f_shift_rate_final:.4f} Hz/s, Acceleration: {f_shift_acc_final:.4e} Hz/s²'
                plt.figtext(0.5, 0.04, info, ha='center', fontsize=params['font_size']-2)
            else:
                info = f'Linear rate: {f_shift_rate_final:.4f} Hz/s'
                plt.figtext(0.5, 0.04, info, ha='center', fontsize=params['font_size']-2)
            
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
        elif poly_degree_final == 3:
            # 三阶多项式拟合情况下：
            # 频移量 = f_shift_rate_final*t + f_shift_acc_final*t^2 + f_shift_rate_acc_final*t^3
            # 相位累积 = f_shift_rate_final*t^2/2 + f_shift_acc_final*t^3/3 + f_shift_rate_acc_final*t^4/4
            # 对于离散序列，t = array_range/fs
            t = array_range/fs
            phase_accumulation = coefs_final[1]*t**2/2 + coefs_final[2]*t**3/3 + coefs_final[3]*t**4/4
            compensation_carrier_final = np.exp(-2j*np.pi*phase_accumulation)
        else:
            logger.warning("poly_degree_final is not 1, 2 or 3, using linear drift correction")
            return wave_compensated_linear, f_shift_rate/fs
            
        # 绘制补偿载波的频谱图
        if debug_plots:
            plt.figure(figsize=(10, 6))
            compensation_carrier_final_db, f, t = calculate_spectrogram(
                compensation_carrier_final, fs, bins_per_tone, steps_per_symbol
            )
            
            plt.imshow(compensation_carrier_final_db, aspect='auto', origin='lower', 
               extent=[t[0], t[-1], f[0], f[-1]])
            plt.colorbar(label='Magnitude (dB)')
            plt.xlabel('Time (s)', fontsize=params['font_size'])
            plt.ylabel('Frequency (Hz)', fontsize=params['font_size'])
            plt.title('Corrected Signal Spectrogram', fontsize=params['font_size']+2)
            plt.grid(True)
            plt.savefig('compensation_carrier_spectrogram.png')
            plt.close()
        
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
        
        logger.info("Final drift parameters: %s", drift_params)
        # 使用poly_final.transform来正确转换输入数据
        first_point = poly_final.transform(regression_x[0].reshape(1, -1))
        last_point = poly_final.transform(regression_x[-1].reshape(1, -1))
        first_pred = model_final.predict(first_point)[0]
        last_pred = model_final.predict(last_point)[0]
        f_shift_rate_real = (first_pred - last_pred)/(regression_x[0]-regression_x[-1]) + f_shift_rate
        return wave_compensated_final, f_shift_rate_real/fs
    
    # 如果数据点不足进行高次拟合，退回到线性拟合结果
    logger.warning("Not enough data for high-order fitting, using linear drift correction")
    return wave_compensated_linear, f_shift_rate/fs
