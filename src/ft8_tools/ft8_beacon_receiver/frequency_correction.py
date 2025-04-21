import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from ..ft8_demodulator.ftx_types import FT8Waterfall

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

def correct_frequency_drift(wave_complex: np.ndarray, fs: float, sym_bin: float, sym_t: float, waterfall: FT8Waterfall, params=None):
    """
    对FT8信号进行频率漂移校正
    
    参数:
    wave_complex: 复数形式的输入信号
    fs: 采样率
    sym_bin: 符号频率间隔
    sym_t: 符号时间长度
    waterfall: FT8Waterfall对象，包含频谱图数据
    params: 可选参数字典，包含以下字段：
        - nsync_sym: 同步符号数量 (默认: 7)
        - ndata_sym: 数据符号数量 (默认: 58)
        - zscore_threshold: Z分数阈值 (默认: 5)
        - max_iteration_num: 最大迭代次数 (默认: 400)
        - debug_plots: 是否生成调试图 (默认: False)
    
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
        'max_iteration_num': 4000,
        'debug_plots': False
    }
    
    if params is None:
        params = default_params
    else:
        for key, value in default_params.items():
            if key not in params:
                params[key] = value

    # 基本变量
    time_osr = waterfall.time_osr
    freq_osr = waterfall.freq_osr
    nsync_sym = params['nsync_sym']
    ndata_sym = params['ndata_sym']
    zscore_threshold = params['zscore_threshold']
    max_iteration_num = params['max_iteration_num']
    debug_plots = True
    
    # 信号长度
    nsamples = len(wave_complex)
    tlength = nsamples / fs

    # 从waterfall中获取频谱图数据
    sx_filtered_db = waterfall.mag
    
    # 计算最大幅度频率-时间序列
    max_freq_indices = np.argmax(sx_filtered_db, axis=0)
    
    # 计算实际频率
    freq_step = sym_bin / freq_osr
    max_freqs = max_freq_indices * freq_step
    
    # 保存去异常前的频率索引用于绘图
    if debug_plots:
        max_freq_indices_before = max_freq_indices.copy()
    
    # 使用Z分数方法识别异常值
    iteration_num = 0
    outlier_indices = np.array([])
    
    while True:
        x = np.arange(len(max_freq_indices)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, max_freq_indices)
        max_freq_indices_fitted = model.predict(x)
        
        residuals = max_freq_indices - max_freq_indices_fitted
        z_scores = np.abs(stats.zscore(residuals))
        outlier_indices = np.where(z_scores > zscore_threshold/2)[0]
        outliers = max_freq_indices[outlier_indices]

        if len(outlier_indices) == 0 or iteration_num == max_iteration_num:
            break

        i = np.argmax(z_scores[outlier_indices])
        sx_filtered_db[outliers[i], outlier_indices[i]] = np.min(sx_filtered_db)
        max_freq_indices[outlier_indices[i]] = np.argmax(sx_filtered_db[:, outlier_indices[i]])

        iteration_num += 1

    # 更新最大频率
    max_freqs = max_freq_indices * freq_step
    
    # 绘制去异常前后的对比图
    if debug_plots:
        # 时间轴，单位为秒
        time_axis = np.arange(len(max_freq_indices)) * sym_t / time_osr
        
        # 计算去异常前后的实际频率
        max_freqs_before = max_freq_indices_before * freq_step
        max_freqs_after = max_freq_indices * freq_step
        
        # 计算去异常前后的线性回归
        x_before = np.arange(len(max_freq_indices_before)).reshape(-1, 1)
        model_before = LinearRegression()
        model_before.fit(x_before, max_freq_indices_before)
        max_freq_indices_fitted_before = model_before.predict(x_before)
        max_freqs_fitted_before = max_freq_indices_fitted_before * freq_step
        
        x_after = np.arange(len(max_freq_indices)).reshape(-1, 1)
        model_after = LinearRegression()
        model_after.fit(x_after, max_freq_indices)
        max_freq_indices_fitted_after = model_after.predict(x_after)
        max_freqs_fitted_after = max_freq_indices_fitted_after * freq_step
        
        # 创建图像
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.scatter(time_axis, max_freqs_before, s=10, c='blue', alpha=0.7, label='Original Frequencies')
        plt.plot(time_axis, max_freqs_fitted_before, 'r-', linewidth=2, label='Linear Fit')
        if len(outlier_indices) > 0:
            plt.scatter(time_axis[outlier_indices], max_freqs_before[outlier_indices], 
                       s=80, facecolors='none', edgecolors='red', label='Detected Outliers')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Frequency-Time Curve Before Outlier Removal')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.scatter(time_axis, max_freqs_after, s=10, c='green', alpha=0.7, label='Corrected Frequencies')
        plt.plot(time_axis, max_freqs_fitted_after, 'r-', linewidth=2, label='Linear Fit')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Frequency-Time Curve After Outlier Removal')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('frequency_correction_before_after.png', dpi=150)
        plt.show()

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
        
    # 相关计算
    sync_correlation = np.correlate(max_freqs, three_sync_correlation_seq, mode='full')

    # TODO: 这里需要优化，因为同步序列相关波形在同步序列结束后的值会影响相关峰值的检测
    sync_correlation[:len(three_sync_correlation_seq) -1] = 0

    correlation_peak_index = np.argmax(sync_correlation)
    correlation_peak_time_block_index = correlation_peak_index - (len(three_sync_correlation_seq) - 1) + samples_per_sym//2

    # 回归分析
    regression_x = np.array([])
    regression_y = np.array([])

    for i in range(3):
        start_idx = i*(nsync_sym+ndata_sym//2)*time_osr + correlation_peak_time_block_index
        end_idx = start_idx + (nsync_sym-1) * time_osr + 1
        x_step = sym_t/time_osr
        regression_x = np.append(regression_x, np.arange(start_idx, min(end_idx, len(max_freqs))) * x_step)
        regression_y = np.append(regression_y, max_freqs[start_idx:min(end_idx, len(max_freqs))])

    regression_x = regression_x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(regression_x, regression_y)
    f_shift_est_k_hz_ps = model.coef_[0]  # 估计的频率漂移率 (Hz/s)
    
    # 转换为每个采样点的频率漂移率
    f_shift_est_k_hz_psample = f_shift_est_k_hz_ps / fs

    # 频率偏移补偿
    compensation_carrier = np.exp(-2j*np.pi*f_shift_est_k_hz_psample*np.arange(nsamples)**2/(2*fs))
    wave_compensated = wave_complex * compensation_carrier

    return wave_compensated, f_shift_est_k_hz_psample
