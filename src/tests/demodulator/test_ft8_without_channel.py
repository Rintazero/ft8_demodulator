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

fs = 10e3
f0 = 550
fc = 0
snr_db = -17

payload = np.random.randint(0, 255, size=10, dtype=np.uint8)
payload[9] &= 0xF8

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
    bins_per_tone=4,
    steps_per_symbol=4,
    max_candidates=20,
    min_score=1,
    max_iterations=20
)

print("--------------------------------")
print("payload: ", [hex(x) for x in payload])
print("results: ", results)
print("message: ", [hex(x) for x in results[0][0].payload])
print("--------------------------------")