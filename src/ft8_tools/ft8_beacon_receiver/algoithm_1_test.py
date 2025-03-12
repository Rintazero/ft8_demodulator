import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# Set matplotlib font settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use DejaVu Sans for better compatibility
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

def gfsk_pulse(bt, t):
    """
    Generate GFSK pulse
    
    Parameters:
    bt: Bandwidth-time product
    t: Time sequence
    
    Returns:
    output: GFSK pulse
    """
    k = np.pi * np.sqrt(2.0/np.log(2.0))
    output = 0.5 * (scipy.special.erf(k*bt*(t+0.5)) - scipy.special.erf(k*bt*(t-0.5)))
    return output

# Basic variables
time_osr = 2
freq_osr = 2
NSyncSym = 7
NDataSym = 58
waterfall_freq_range = (500, 3050)
ZscoreThreshold = 5
MaxIterationNum = 400
SNR = -10

# Load raw data
rawSignalInfo = scipy.io.loadmat("src/ft8_tools/ft8_beacon_receiver/data/raw/waveInfo.mat")
fs = rawSignalInfo.get("Fs")[0][0]
f0 = rawSignalInfo.get("F0")[0][0]
waveRe = rawSignalInfo.get("waveRe")[0]
waveIm = rawSignalInfo.get("waveIm")[0]
nsamples = len(waveRe)
wave = np.complex64(waveRe + waveIm)
SymBin = rawSignalInfo.get("SymBin")[0][0]
SymT = rawSignalInfo.get("SymT")[0][0]

# Construct (linear) frequency shift carrier
fShift_t0_Hz = 0.0  # Initial frequency offset
fShift_k_Hzpsample = 2250/nsamples  # Frequency shift rate
shiftCarrier = np.exp(2j*np.pi*fShift_t0_Hz*np.arange(nsamples)/fs + 
                     2j*np.pi*fShift_k_Hzpsample*np.arange(nsamples)**2/(2*fs))

# Multiply frequency shift carrier with original signal
wave_shift = (waveRe + (1j) * waveIm) * shiftCarrier

# Add Gaussian noise
wave_power = np.mean(np.abs(wave)**2)
noise_power = wave_power / (10**(SNR/10))
noise = np.random.normal(0, np.sqrt(noise_power), nsamples)
wave_noise = np.complex64(wave_shift + noise)

# STFT analysis
nsps = int(SymT*fs)
nfft = int(fs*SymT*freq_osr)
tlength = len(wave_noise) / fs
hann_window = scipy.signal.windows.hann(M=nfft, sym=False)
SFT = scipy.signal.ShortTimeFFT(win=hann_window, hop=nsps//time_osr, fs=fs, fft_mode="onesided")
sx = SFT.stft(np.real(wave_noise))

# Frequency range filtering
freqs = np.linspace(0, fs/2, sx.shape[0])
freq_mask = (freqs >= waterfall_freq_range[0]) & (freqs <= waterfall_freq_range[1])
sx_filtered = sx[freq_mask, :]
sx_filtered_real = np.real(sx_filtered)
sx_filtered_imag = np.imag(sx_filtered)
sx_filtered_power = sx_filtered_real**2 + sx_filtered_imag**2
sx_filtered_db = 10 * np.log10(sx_filtered_power)

# Calculate maximum amplitude frequency-time sequence
windowSum = np.zeros(sx_filtered_db.shape)
for i in range(sx_filtered_db.shape[0]):
    for j in range(sx_filtered_db.shape[1]):
        if i < sx_filtered_db.shape[0] - freq_osr:
            windowSum[i][j] = np.sum(sx_filtered_db[i:i+freq_osr,j])
        else:
            windowSum[i][j] = np.sum(sx_filtered_db[i:,j])
windowIndices = np.argmax(windowSum, axis=0)
max_freq_indices = np.zeros(sx_filtered_db.shape[1])
for i in range(sx_filtered_db.shape[1]):
    max_freq_indices[i] = windowIndices[i] + np.argmax(sx_filtered_db[windowIndices[i]:windowIndices[i]+freq_osr,i])
max_freq_indices = max_freq_indices.astype(int)

max_freqs = waterfall_freq_range[0] + freqs[max_freq_indices]

# Plot max_freqs before processing
plt.figure(num=0, figsize=(10, 6))
plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-', color='blue', label='Original Data')
plt.title('Maximum Frequency vs Time (Before Processing)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(True)
plt.legend()
plt.xlim(0, tlength)
plt.ylim(waterfall_freq_range[0], waterfall_freq_range[1])
plt.savefig('max_frequencies_before_processing.png')
plt.show()

# Identify outliers using Z-score method
iterationNum = 0
while True:
    x = np.arange(len(max_freq_indices)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, max_freq_indices)
    max_freq_indices_fitted = model.predict(x)
    
    residuals = max_freq_indices - max_freq_indices_fitted
    z_scores = np.abs(stats.zscore(residuals))
    outlier_indices = np.where(z_scores > ZscoreThreshold)[0]
    outliers = max_freq_indices[outlier_indices]

    if len(outlier_indices) == 0 or iterationNum == MaxIterationNum:
        break

    i = np.argmax(z_scores[outlier_indices])
    sx_filtered_db[outliers[i],outlier_indices[i]] = np.min(sx_filtered_db)
    windowIndices[outlier_indices[i]] = np.argmax(sx_filtered_db[:,outlier_indices[i]])
    max_freq_indices[outlier_indices[i]] = windowIndices[outlier_indices[i]] + np.argmax(sx_filtered_db[windowIndices[outlier_indices[i]]:windowIndices[outlier_indices[i]]+freq_osr,outlier_indices[i]])

    iterationNum += 1

max_freqs = waterfall_freq_range[0] + freqs[max_freq_indices]

# Plot max_freqs after processing
plt.figure(num=6, figsize=(10, 6))
plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-', color='blue', label='Processed Data')
plt.title('Maximum Frequency vs Time (After Processing)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(True)
plt.legend()
plt.xlim(0, tlength)
plt.ylim(waterfall_freq_range[0], waterfall_freq_range[1])

# Mark outliers in red
plt.scatter(outlier_indices * SymT/time_osr, max_freqs[outlier_indices], color='red', marker='x', s=100, label='Outliers')
plt.legend()

plt.savefig('max_frequencies_after_processing.png')
plt.show()


# Construct synchronization sequence
SyncSeq = (np.array([3,1,4,0,6,5,2]) + 1)
SyncSeq = SyncSeq - np.mean(SyncSeq)

# GFSK pulse shaping for each symbol
samples_per_sym = time_osr * 2
t_pulse = np.linspace(-1, 1, samples_per_sym+1)
gfsk_shape = gfsk_pulse(bt=2.0, t=t_pulse)

# Extend sync sequence length to accommodate shaped pulse
syncCorrelationSeq = np.zeros((NSyncSym-1) * time_osr + samples_per_sym + 1)
# 可视化 syncCorrelationSeq
plt.figure(num=7, figsize=(10, 6))
plt.plot(syncCorrelationSeq, marker='o', linestyle='-', color='b')
plt.title('同步序列相关波形')
plt.xlabel('采样点')
plt.ylabel('幅值')
plt.grid(True)
plt.xlim(0, len(syncCorrelationSeq))
plt.ylim(np.min(syncCorrelationSeq)-0.5, np.max(syncCorrelationSeq)+0.5)
plt.savefig('sync_correlation_seq.png')
plt.show()


# Pulse shaping for each sync symbol
for sym_idx in range(NSyncSym):
    syncCorrelationSeq[sym_idx * time_osr:(sym_idx * time_osr) + samples_per_sym + 1] += gfsk_shape * SyncSeq[sym_idx]

threeSyncCorrelationSeq = np.zeros((3*NSyncSym + NDataSym - 1) * time_osr + 1 + samples_per_sym)

for i in range(3):
    start_idx = i*(NSyncSym+NDataSym//2)*time_osr
    end_idx = start_idx + len(syncCorrelationSeq)
    threeSyncCorrelationSeq[start_idx:end_idx] = syncCorrelationSeq

syncCorrelation = np.correlate(max_freqs, threeSyncCorrelationSeq, mode='full')
correlationPeakIndex = np.argmax(syncCorrelation)
correlationPeakTimeBlockIndex = correlationPeakIndex - (len(threeSyncCorrelationSeq) - 1) + samples_per_sym//2

# Regression analysis
regression_x = np.array([])
regression_y = np.array([])
model = LinearRegression()

for i in range(3):
    start_idx = i*(NSyncSym+NDataSym//2)*time_osr + correlationPeakTimeBlockIndex
    end_idx = start_idx + (NSyncSym-1) * time_osr + 1
    x_step = SymT/time_osr
    regression_x = np.append(regression_x, np.arange(start_idx, min(end_idx, len(max_freqs))) * x_step)
    regression_y = np.append(regression_y, max_freqs[start_idx:end_idx])

regression_x = regression_x.reshape(-1, 1)
model.fit(regression_x, regression_y)
fShift_est_k_Hzps = model.coef_[0]


fShift_k_Hzps = fShift_k_Hzpsample * fs  # 频偏变化率 (Hz/s)
print("fShift_k_Hzps:",fShift_k_Hzps)

print("Linear Frequency Deviation Rate Error:{}Hz".format((fShift_k_Hzps - fShift_est_k_Hzps)*nsamples / fs))

# # 可视化 syncCorrelationSeq
# plt.figure(figsize=(10, 6))
# plt.plot(syncCorrelationSeq, marker='o', linestyle='-', color='b')
# plt.title('同步序列相关波形')
# plt.xlabel('采样点')
# plt.ylabel('幅值') 
# plt.grid()
# plt.xlim(0, len(syncCorrelationSeq))
# plt.ylim(np.min(syncCorrelationSeq)-0.5, np.max(syncCorrelationSeq)+0.5)
# plt.savefig('sync_correlation_seq.png')

fShift_est_k_Hzpsample = fShift_est_k_Hzps / fs

# Frequency offset compensation
CompensationCarrier = np.exp(-2j*np.pi*fShift_est_k_Hzpsample*np.arange(nsamples)**2/(2*fs))
wave_compensated = wave_noise * CompensationCarrier
wave_compensated_sx = SFT.stft(np.real(wave_compensated))
compensated_freq_range = (waterfall_freq_range[0], waterfall_freq_range[0]+200)
compensated_freq_mask = (freqs >= compensated_freq_range[0]) & (freqs <= compensated_freq_range[1])
wave_compensated_sx_filtered = wave_compensated_sx[compensated_freq_mask, :]
wave_compensated_sx_filtered_real = np.real(wave_compensated_sx_filtered)
wave_compensated_sx_filtered_imag = np.imag(wave_compensated_sx_filtered)
wave_compensated_sx_filtered_power = wave_compensated_sx_filtered_real**2 + wave_compensated_sx_filtered_imag**2
wave_compensated_sx_filtered_db = 10 * np.log10(wave_compensated_sx_filtered_power)

# Visualization
plt.figure(num=1, figsize=(10, 6))
plt.imshow(sx_filtered_db, aspect='auto', origin='lower', 
           extent=[0, tlength, waterfall_freq_range[0], waterfall_freq_range[1]])
plt.colorbar(label='Magnitude(dB)')
plt.title('Short-Time Fourier Transform (STFT) Magnitude(dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.grid()
plt.savefig('stft_magnitude.png')
plt.show()

plt.figure(num=2, figsize=(10, 6))
plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-', zorder=1)
plt.title('Maximum Frequencies Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.xlim(0, tlength)
plt.ylim(waterfall_freq_range[0], waterfall_freq_range[1])
highlight_index = correlationPeakTimeBlockIndex + list(range(NSyncSym*time_osr))
plt.scatter(highlight_index * SymT/time_osr, max_freqs[highlight_index], color='red', marker='o', zorder=2)
plt.scatter(outlier_indices * SymT/time_osr, max_freqs[outlier_indices], color='green', marker='o', zorder=2)
plt.grid()
plt.savefig('max_frequencies.png')
plt.show()

plt.figure(num=3, figsize=(10, 6))
plt.plot(syncCorrelation, marker='o', linestyle='-', color='b')
plt.title('Synchronization Correlation')
plt.xlabel('Sample Index')
plt.ylabel('Correlation Value')
plt.grid()
plt.xlim(0, len(syncCorrelation))
plt.ylim(np.min(syncCorrelation), np.max(syncCorrelation))
plt.savefig('sync_correlation.png')
plt.show()

plt.figure(num=4, figsize=(10, 6))
plt.imshow(wave_compensated_sx_filtered_db, aspect='auto', origin='lower', 
           extent=[0, tlength, compensated_freq_range[0], compensated_freq_range[1]])
plt.colorbar(label='Magnitude (dB)')
plt.title('Wave Compensated STFT Magnitude (Filtered)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.grid()
plt.savefig('wave_compensated_stft.png')
plt.show()

plt.figure(num=5, figsize=(10, 6))
plt.scatter(regression_x, regression_y, color='blue', alpha=0.5, label='Original Data Points')
plt.plot(regression_x, model.predict(regression_x), color='red', label='Fitted Line')
plt.title('Frequency Offset Regression Analysis')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(True)
plt.legend()
plt.savefig('regression_analysis2.png')
plt.show()
