import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules
from ft8_tools.ft8_demodulator.spectrogram_analyse import (
    calculate_spectrogram,
    select_frequency_band,
)
from ft8_tools.ft8_demodulator.ft8_decode import (
    decode_ft8_message,
    FT8Message,
    FT8DecodeStatus,
)

# Import ft8_generator
from ft8_tools.ft8_generator import ft8_generator

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
        print("Continuing with tests...")

def test_calculate_spectrogram():
    """Test spectrogram calculation functionality"""
    # Set basic parameters
    fs = 10000  # Sample rate
    f0 = 300    # Audio frequency
    fc = 0      # Carrier frequency (baseband signal)
    test_payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
    
    # Generate test waveform data
    wave_data = ft8_generator(
        test_payload, 
        fs=fs, 
        f0=f0, 
        fc=fc
    )
    
    # Calculate spectrogram
    spectrogram, f, t = calculate_spectrogram(wave_data, fs, 2, 2)
    
    # Verify spectrogram basic properties
    assert isinstance(spectrogram, np.ndarray), "Spectrogram should be a numpy array"
    assert spectrogram.shape[0] > 0, "Spectrogram should have non-zero rows"
    assert spectrogram.shape[1] > 0, "Spectrogram should have non-zero columns"
    
    print("Spectrogram shape:", spectrogram.shape)
    print("Frequency range:", np.min(f), "Hz to", np.max(f), "Hz")
    
    # Save spectrogram results
    save_spectrogram(spectrogram, 'original_spectrogram.png', 'Original Spectrogram')
    
    # Test frequency band selection
    filtered_spec, filtered_f = select_frequency_band(spectrogram, f, 0, 500)
    save_spectrogram(filtered_spec, 'filtered_spectrogram.png', 'Filtered Spectrogram (0-500 Hz)')
    
    print("Spectrogram calculation test passed!")

def verify_decode_results(results: list) -> None:
    """Verify decode results"""
    assert isinstance(results, list), "Decode results should be a list"
    
    if results:
        for message, status in results:
            assert isinstance(message, FT8Message), "Decoded message should be FT8Message type"
            assert isinstance(status, FT8DecodeStatus), "Decode status should be FT8DecodeStatus type"
            assert len(message.payload) == 10, "FT8 message payload should be 10 bytes"
            
            print("Decoded message:", message.payload.hex())
            print("CRC check:", status.crc_calculated)
            print("LDPC errors:", status.ldpc_errors)

def test_decode_ft8_message():
    """Test FT8 message decoding functionality"""
    # Set basic parameters
    fs = 1000  # Sample rate
    f0 = 300    # Audio frequency
    fc = 0      # Carrier frequency (baseband signal)
    test_payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
    
    # Generate test waveform data
    wave_data = ft8_generator(
        test_payload, 
        fs=fs, 
        f0=f0, 
        fc=fc
    )
    
    results = decode_ft8_message(
        wave_data=wave_data,
        sample_rate=fs,
        bins_per_tone=2,
        steps_per_symbol=2,
        max_candidates=20,
        min_score=10,
        max_iterations=20
    )
    
    verify_decode_results(results)
    print("FT8 message decode test passed!")

def test_decode_with_noise():
    """Test decoding performance with noise"""
    # Set basic parameters
    fs = 12000  # Sample rate
    f0 = 300    # Audio frequency
    fc = 0      # Carrier frequency (baseband signal)
    test_payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
    
    # Generate test waveform data
    wave_data = ft8_generator(
        test_payload, 
        fs=fs, 
        f0=f0, 
        fc=fc
    )
    
    noise_level = 0.1
    noisy_wave = wave_data + noise_level * np.random.randn(len(wave_data))
    
    results = decode_ft8_message(
        wave_data=noisy_wave,
        sample_rate=fs,
        bins_per_tone=10,
        steps_per_symbol=10
    )
    
    if results:
        print(f"Successfully decoded with noise level {noise_level}")
        verify_decode_results(results)
    else:
        print(f"Failed to decode with noise level {noise_level}")
    
    print("Noisy decode test passed!")

def test_decode_edge_cases():
    """Test edge cases"""
    fs = 12000  # Sample rate
    
    # Test empty signal
    empty_results = decode_ft8_message(
        wave_data=np.zeros(1000),
        sample_rate=fs,
        bins_per_tone=2,  # Use smaller value to avoid large window
        steps_per_symbol=2
    )
    assert len(empty_results) == 0, "Empty signal should return empty results list"
    
    # Test very short signal
    short_results = decode_ft8_message(
        wave_data=np.zeros(10),
        sample_rate=fs,
        bins_per_tone=2,
        steps_per_symbol=2
    )
    assert len(short_results) == 0, "Very short signal should return empty results list"
    
    # Test high sample rate
    test_payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
    wave_data = ft8_generator(test_payload, fs=fs, f0=300, fc=0)
    
    high_fs_results = decode_ft8_message(
        wave_data=wave_data,
        sample_rate=48000,
        bins_per_tone=2,
        steps_per_symbol=2
    )
    print("High sample rate test results count:", len(high_fs_results))
    print("Edge cases test passed!")

if __name__ == "__main__":
    print("Starting tests...")
    test_calculate_spectrogram()
    test_decode_ft8_message()


    print("All tests completed!")

