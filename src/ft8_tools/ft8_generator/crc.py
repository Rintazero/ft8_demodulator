import numpy as np
from numpy.typing import NDArray

FT8_CRC_BITS = 14
FT8_CRC_PADDED_BITS = 96
FT8_PAYLOAD_BITS = 77
FT8_CRC_POLYNOMIAL = np.uint16(0x2757) # CRC-14 polynomial without the leading (MSB) 1

def calc_crc(msg: NDArray[np.uint8], num_bits: int) -> np.uint16:
    remainder = np.uint16(0x0000)
    idx_byte = 0
    TOP_BIT = np.uint16(0x0001) << (FT8_CRC_BITS - 1)

    for idx_bit in range(num_bits):
        if idx_bit % 8 == 0:
            remainder ^= (np.uint16(msg[idx_byte]) << (FT8_CRC_BITS - 8))
            idx_byte += 1
        if remainder & TOP_BIT != 0:
            remainder = (remainder << 1) ^ FT8_CRC_POLYNOMIAL
        else:
            remainder = (remainder << 1)
    return np.uint16(remainder & ((TOP_BIT << 1) - 0x0001))


def crc_generator(payload_10bytes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # 根据 "The FT4 and FT8 Communication Protocols" 的描述，a91_12bytes的结构如下:
    # a91_12bytes(96 bits) = payload_10bytes(77 bits) + zeros(5 bits) + crc(14 bits)
    # 协议中添加 5 bits 的 zeros, 是为了将数据拓展至 96 bits(整bytes), 以便于直接调用 C 语言 crc 库

    a91_12bytes = np.zeros(12, dtype=np.uint8)
    # copy payload_10bytes to a91_12bytes
    for i in range(10):
        a91_12bytes[i] = payload_10bytes[i]

    a91_12bytes[9] &= 0xF8 # clear the last 3 bits, then the first 77 bits are the payload
    a91_12bytes[10] = 0x00
    a91_12bytes[11] = 0x00

    # 计算 crc (数据位为 payload(77 bits) + 5 bits zeros; 多项式为 0x6757(依协议))
    checksum = calc_crc(a91_12bytes, FT8_CRC_PADDED_BITS - FT8_CRC_BITS)

    # 将 crc 值写入 a91_12bytes (crc_14bits 直接写在 payload_77bits 之后 (忽略 5bits zeros) )
    a91_12bytes[9] |= np.uint8(checksum >> 11)
    a91_12bytes[10] = np.uint8(checksum >> 3)
    a91_12bytes[11] = np.uint8(checksum << 5)

    return a91_12bytes

def get_crc_from_a91(a91_12bytes: NDArray[np.uint8]) -> np.uint16:
    checksum = np.uint16((np.uint16(a91_12bytes[9] & 0x07) << 11) | (np.uint16(a91_12bytes[10]) << 3) | (np.uint16(a91_12bytes[11]) >> 5))
    return checksum

def calc_crc_from_a91(a91_12bytes: NDArray[np.uint8]) -> np.uint16:
    msg = np.zeros(12, dtype=np.uint8)
    for i in range(10):
        msg[i] = a91_12bytes[i]
    msg[9] &= 0xF8
    msg[10] = 0x00
    msg[11] = 0x00

    checksum = calc_crc(msg, FT8_CRC_PADDED_BITS - FT8_CRC_BITS)
    return checksum

def check_crc(a91_12bytes: NDArray[np.uint8]) -> bool:
    checksum_A = calc_crc_from_a91(a91_12bytes)
    checksum_B = get_crc_from_a91(a91_12bytes)
    return checksum_A == checksum_B
