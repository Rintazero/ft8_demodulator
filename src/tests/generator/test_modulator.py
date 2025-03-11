import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import sys
import os
import unittest

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ft8_tools.ft8_generator import crc, ldpc, encoder, modulator

__name__ = "__main__"

class TestModulator(unittest.TestCase):
    def test_encoder(self):
        payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
        itones = encoder.ft8_encode(payload)
        freq_seq = modulator.gfsk_modulation_waveform_generator(itones,1e3)
        plt.figure(figsize=(10, 6))
        plt.plot(freq_seq, label='Frequency (Hz)')
        plt.title('Frequency Visualization of FT8 Waveform')
        plt.xlabel('Sample Index')
        plt.ylabel('Frequency (Hz)')
        plt.show()

    def test_ft8_modulation_waveform_generator(self):
        payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
        itones = encoder.ft8_encode(payload)
        gfsk_waveform = modulator.gfsk_modulation_waveform_generator(itones,10e3)
        ft8_waveform = modulator.ft8_modulation_waveform_generator(gfsk_waveform,10e3,200)
        plt.figure(figsize=(10, 6))
        plt.plot(np.real(ft8_waveform), label='Real Part')
        plt.plot(np.imag(ft8_waveform), label='Imaginary Part')
        plt.title('Frequency Visualization of FT8 Waveform')
        plt.xlabel('Sample Index')
        plt.ylabel('Frequency (Hz)')
        plt.show()
        wav_data = (np.real(ft8_waveform) * 32767).astype(np.int16)
        sci.io.wavfile.write('ft8_waveform.wav', 10000, wav_data)

if __name__ == '__main__':
    unittest.main()