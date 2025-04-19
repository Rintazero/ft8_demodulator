import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules
from ft8_tools.ft8_generator import (
    ft8_generator,
    crc_generator,
    ldpc_generator,
)
from ft8_tools.ft8_demodulator.spectrogram_analyse import (
    calculate_spectrogram,
    select_frequency_band,
)
from ft8_tools.ft8_demodulator.ft8_decode import (
    decode_ft8_message,
    FT8Message,
    FT8DecodeStatus,
)
from ft8_tools.ft8_demodulator.ldpc_decoder import (
    ldpc_check,
    bp_decode,
)

# tools functions
def payload_crc_ldpc_22bytes_to_174bits(payload_crc_ldpc_22bytes: np.ndarray) -> np.ndarray:
    payload_crc_ldpc_bits = np.zeros(174, dtype=np.uint8)
    for i in range(len(payload_crc_ldpc_22bytes)):
        byte = payload_crc_ldpc_22bytes[i]
        for j in range(8):
            if i * 8 + j < 174:  # 确保不超过174位
                payload_crc_ldpc_bits[i * 8 + j] = (byte >> (7 - j)) & 1
    return payload_crc_ldpc_bits

def verify_decode_results(results: list, payload: np.ndarray) -> None:
    """Verify decode results"""

def test_step(payload: np.ndarray, fs: int, f0: int, fc: int, snr_db: int) -> bool:
    """Test step"""
    wave_data = ft8_generator(
        payload, 
        fs=fs, 
        f0=f0, 
        fc=fc
    )
    signal_power = np.mean(wave_data**2)  # Calculate the signal power
    noise_power = signal_power / (10**(snr_db / 10))  # Calculate the noise power based on SNR
    noise = np.sqrt(noise_power) * np.random.randn(len(wave_data))  # Generate noise
    noisy_wave_data = wave_data + noise  # Add noise to the waveform data
    wave_data = noisy_wave_data
    results = decode_ft8_message(
        wave_data=wave_data,
        sample_rate=fs,
        bins_per_tone=2,
        steps_per_symbol=2,
        max_candidates=20,
        min_score=1,
        max_iterations=20
    )
    if len(results) > 0:
        return True
    else:
        return False

fs_start_hz = 2000
fs_end_hz = 10000 + 500
fs_step_hz = 500

snr_db_start = -21
snr_db_end = -10
snr_db_step = 0.2

success_ratio_threshold = 0.5
num_rounds_per_test_point = 20

results = np.zeros((fs_end_hz - fs_start_hz) // fs_step_hz)
results = results + 3
fs_index = 0
for fs in range(fs_start_hz, fs_end_hz, fs_step_hz):
    print(f"Testing fs: {fs} Hz")
    for snr_db in np.arange(snr_db_start, snr_db_end, snr_db_step):
        print(f"    Testing snr_db: {snr_db} dB, @fs: {fs} Hz")
        success_count = 0
        fail_count = 0
        for i in range(num_rounds_per_test_point):
            payload = np.random.randint(0, 256, size=10, dtype=np.uint8)
            success = test_step(payload, fs, 0, 0, snr_db)
            if success:
                success_count += 1
            else:
                fail_count += 1
            if fail_count > num_rounds_per_test_point * (1 - success_ratio_threshold):
                break
        success_ratio = success_count / num_rounds_per_test_point
        if success_ratio >= success_ratio_threshold:
            results[fs_index] = snr_db
            break
    fs_index += 1



# Create frequency array for x-axis
frequencies = np.arange(fs_start_hz, fs_end_hz, fs_step_hz)
    
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(frequencies/2, results, 'b-o', linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Bandwidth (Hz)')
plt.ylabel('SNR (dB)')
plt.title('Minimum SNR Required for Successful Decoding')
plt.xticks(frequencies/2)
plt.yticks(np.arange(snr_db_start, snr_db_end + 1, 2))
    
# Save the plot
plt.savefig('snr_vs_frequency_20250417_1308.png')
plt.close()
plt.show()
























