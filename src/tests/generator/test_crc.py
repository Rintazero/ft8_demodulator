import numpy as np
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ft8_tools.ft8_generator import crc

__name__ = "__main__"

class TestCRCGenerator(unittest.TestCase):
    
    def test_crc_generation(self):
        payload = np.zeros(10, dtype=np.uint8)
        # payload[9] = 0xF8
        # payload[8] = 0x09
        # payload[7] = 0x08
        # payload[6] = 0x07
        # payload[5] = 0x06
        # payload[4] = 0x05
        # payload[3] = 0x04
        # payload[2] = 0x03
        # payload[1] = 0x02
        # payload[0] = 0xAA

        payload[9] = 0x51
        payload[8] = 0x94
        payload[7] = 0xE3
        payload[6] = 0xA1
        payload[5] = 0x07
        payload[4] = 0xE2   
        payload[3] = 0x6A
        payload[2] = 0x8A
        payload[1] = 0x3F
        payload[0] = 0x1C

        a91_12bytes = np.zeros(12, dtype=np.uint8)

        a91_12bytes = crc.crc_generator(payload)

        np.set_printoptions(formatter={'int': lambda x: format(x, '02X')})
        print(a91_12bytes)
        
        # Replace the expected_crc with the actual expected value based on your implementation
        #expected_crc = 0xXX  # Replace with the expected CRC value
        #self.assertEqual(crc, expected_crc, "CRC value does not match the expected value.")

    def test_crc_check(self):
        payload = np.zeros(10, dtype=np.uint8)
        payload[9] = 0x51
        payload[8] = 0x94
        payload[7] = 0xE3
        payload[6] = 0xA1
        payload[5] = 0x07
        payload[4] = 0xE2   
        payload[3] = 0x6A
        payload[2] = 0x8A
        payload[1] = 0x3F
        payload[0] = 0x1C
        a91_12bytes = np.zeros(12, dtype=np.uint8)
        a91_12bytes = crc.crc_generator(payload)
        print(crc.check_crc(a91_12bytes))
        

if __name__ == '__main__':
    unittest.main()


