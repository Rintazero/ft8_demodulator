import numpy as np
import unittest
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ft8_tools.ft8_generator import crc, ldpc, encoder

__name__ = "__main__"

class TestEncoder(unittest.TestCase):
    def test_encoder(self):
        payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
        np.set_printoptions(formatter={'int': lambda x: format(x, '02X')})
        print("payload: ", payload)
        a91_12bytes = crc.crc_generator(payload)
        print("a91_12bytes: ", a91_12bytes)
        codeword_22bytes = ldpc.ldpc_generator(a91_12bytes)
        print("codeword_22bytes: ", codeword_22bytes)

        np.set_printoptions(formatter={'int': lambda x: format(x, 'd')})
        symbolIdSequence = encoder.symbolIdSequence_generator(codeword_22bytes)
        print("symbolIdSequence: ", symbolIdSequence)

        itones = encoder.itones_generator(symbolIdSequence)
        print("itones: ", itones)

    def test_ft8_encode(self):
        payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
        itones = encoder.ft8_encode(payload)
        print("ft8_encode-itones: ", itones)

if __name__ == '__main__':
    unittest.main()