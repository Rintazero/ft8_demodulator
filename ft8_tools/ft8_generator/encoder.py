import numpy as np
from numpy.typing import NDArray
from .ldpc import FT8_LDPC_N as FT8_CODEWORD_BITS
from . import crc
from . import ldpc

FT8_DATA_SYMBOL_NUM = FT8_CODEWORD_BITS // 3
FT8_COSTAS_SYMBOL_NUM = 7
FT8_COSTAS_SEQ_NUM = 3
FT8_SYMBOL_NUM = FT8_DATA_SYMBOL_NUM + FT8_COSTAS_SYMBOL_NUM * FT8_COSTAS_SEQ_NUM

FT8_GRAY_MAP = np.array([0, 1, 3, 2, 5, 6, 4, 7], dtype=np.uint8)
FT8_COSTAS_PATTERN = np.array([3, 1, 4, 0, 6, 5, 2], dtype=np.uint8)

def symbolIdSequence_generator(codeword: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Generate the symbol ID sequence for the given codeword.
    """
    symbolIdSequence = np.zeros(FT8_DATA_SYMBOL_NUM, dtype=np.uint8)
    bit_mask = np.uint8(0x80)
    idx_byte = 0
    idx_sym = 0
    symIdBeforeGray = np.uint8(0x00)
    for idx_bit in range(FT8_CODEWORD_BITS):
        symIdBeforeGray = symIdBeforeGray << 1

        if (codeword[idx_byte] & bit_mask) != 0:
            symIdBeforeGray += np.uint8(0x01)
        if (idx_bit + 1) % 8 == 0:
            bit_mask = np.uint8(0x80)
            idx_byte += 1
        else:
            bit_mask = bit_mask >> 1
        if (idx_bit + 1) % 3 == 0:
            symbolIdSequence[idx_sym] = FT8_GRAY_MAP[symIdBeforeGray]
            symIdBeforeGray = 0
            idx_sym += 1
        
    return symbolIdSequence

def itones_generator(symbolIdSequence: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Generate the ITONES sequence for the given symbol ID sequence.
    """
    itones = np.zeros(FT8_SYMBOL_NUM, dtype=np.uint8)
    idx_sym = 0
    for i in range(FT8_COSTAS_SYMBOL_NUM):
        itones[idx_sym] = FT8_COSTAS_PATTERN[i]
        idx_sym += 1
    for i in range(FT8_DATA_SYMBOL_NUM//2):
        itones[idx_sym] = symbolIdSequence[i]
        idx_sym += 1
    for i in range(FT8_COSTAS_SYMBOL_NUM):
        itones[idx_sym] = FT8_COSTAS_PATTERN[i]
        idx_sym += 1
    for i in range(FT8_DATA_SYMBOL_NUM//2):
        itones[idx_sym] = symbolIdSequence[i + FT8_DATA_SYMBOL_NUM//2]
        idx_sym += 1
    for i in range(FT8_COSTAS_SYMBOL_NUM):
        itones[idx_sym] = FT8_COSTAS_PATTERN[i]
        idx_sym += 1
    return itones

def ft8_encode(payload: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Encode the payload for FT8.
    """
    a91_12bytes = crc.crc_generator(payload)
    codeword_22bytes = ldpc.ldpc_generator(a91_12bytes)
    symbolIdSequence = symbolIdSequence_generator(codeword_22bytes)
    itones = itones_generator(symbolIdSequence)
    return itones

